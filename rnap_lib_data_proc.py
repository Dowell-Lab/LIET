import numpy as np
import copy
from collections import defaultdict

## GENE ANNOTATION HANDLING ###################################################

def annotation(chrom=None, start=None, stop=None, strand=None):
    
    strand_dict = {'+': 1, '-': -1}
    
    out = {
        'chrom': str(chrom),
        'start': int(start),
        'stop': int(stop),
        'strand': strand_dict[strand]
    }
    return out



def annot_loader(annot_file):
    '''
    Loads annotations from basic gtf file into a dictionary and then sorts 
    them by genomic coordinate and strand. Returns sorted dict of form:
    {'chr#': {(start, stop, strand): gene_id, ...}, ...}
    '''

    chromosomes = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chr23', 'chrX', 'chrY'
    ]
    annot = {ch:{} for ch in chromosomes}
    strand_dict = {'+': 1, '-': -1}

    # Open annotation file (tab delimited: chr, start, stop, strand, id)
    with open(annot_file, 'r') as f:

        for i, line in enumerate(f):
            line = line.strip().split('\t')

            if len(line) != 5:
                raise ValueError(f"Annotation at line {i} is incorrectly "
                    f"formatted. See: '{line}'")

            try:
                chrom = str(line[0])
                start = int(line[1])
                stop = int(line[2])
                strnd = strand_dict[line[3]]
                id = str(line[4])

                annot[chrom][(start, stop, strnd)] = id
            except:
                raise ValueError(f"Annotation at line {i} is incorrectly "
                    f"formatted. See: '{line}'")

    # Sort the regions of interest for each chromosome and write to out dict
    annot_sorted = {}
    for ch, rois in annot.items():
        if rois != {}:
            annot_sorted[ch] = {r:annot[ch][r] for r in sorted(rois.keys())}
        else:
            continue

    return annot_sorted



## CONFIG FILE HANDLING #######################################################

def config_loader(config_file):
    '''
    Loads input parameters for model fitting. This includes model scope, 
    prior, data processing, fitting, and results formatting parameters.
    '''

    def float_int_cast(val):
        if '.' in val:
            try:
                return float(val)
            except:
                raise ValueError(f"Input {val} is not a float value.")
        else:
            try:
                return int(val)
            except:
                raise ValueError(f"Input {val} is not an integer value.")

    def bool_cast(val):
        if val == 'True':
            return True
        elif val == 'False':
            return False
        else:
            raise ValueError(f"Input {val} is not a boolean value.")

    prior_names = ['mL', 'sL', 'tI', 'mT', 'sT', 'w', 'mL_a', 'sL_a', 'tI_a']

    config = {
        'FILES': {},
        'MODEL': {}, 
        'PRIORS': {}, 
        'DATA_PROC': {}, 
        'FITTING': {}, 
        'RESULTS': {}
    }

    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            elif line[0] == '[':
                category = line.strip('[]')
                continue
            else:                
                try:
                    line = line.split('=')
                    line = [e.strip() for e in line]
                except:
                    raise ValueError(f"Incorrectly formatted input parameter. "
                        f"See: '{line}'")

            pname = str(line[0])
            pval = str(line[1])

            if pname in prior_names:
                #pval = [e.strip() for e in pval.strip().split(',')]
                #pval = [pval[0], [float_int_cast(e) for e in pval[1:]]]
                #config[category][pname] = pval

                pval = [e.strip() for e in pval.strip().split(',')]         #### NEED TO DO SOMETHING HERE TO HAND IF pval IS None
                
                pval_dict = {}
                for p in pval:
                    p = p.strip().split(':')
                    if p[0] == 'dist':
                        pval_dict[p[0]] = p[1]
                    else:
                        pval_dict[p[0]] = float_int_cast(p[1])
                config[category][pname] = pval_dict

            elif (pname == 'ANNOTATION' or 
                pname == 'BEDGRAPH' or
                pname == 'RESULTS'):
                config[category][pname] = pval

            elif pname == 'ANTISENSE':
                config[category][pname] = bool_cast(pval)
            elif pname == 'BACKGROUND':
                config[category][pname] = bool_cast(pval)

            elif pname == 'RANGE_SHIFT':
                config[category][pname] = bool_cast(pval)
            elif pname == 'PAD':
                config[category][pname] = float_int_cast(pval)

            elif pname == 'ITERATIONS':
                pval = float_int_cast(pval)
                if isinstance(pval, int):
                    config[category][pname] = pval
                else:
                    raise ValueError(f"Input {pname} must be an integer.")
            elif pname == 'LEARNING_RATE':
                pval = float_int_cast(pval)
                if isinstance(pval, float):
                    config[category][pname] = pval
                else:
                    raise ValueError(f"Input {pname} must be a float.")
            else:
                raise ValueError(f"Incorrect input parameter: {pname}")
    
    return config


## PRIORS HANDLING ############################################################

def prior_config(priors, tss, tts):
    '''
    This function formats the priors dict generated by config_loader() so that 
    it conforms to the correct format to be loaded into a LIET model instance. 
    The input 'priors' is still in string format and must be evaluated for any 
    annotation offsets then converted to float/int. The two reference points 
    must be range shifted if RANGE_SHIFT=True in config
    
    NOTE: This is a silly function, but I'm currently keeping it separate in 
    the event that I change the way the config/initialization process is done.
    '''
    #absolute_priors = ['sL', 'tI', 'sT', 'w', 'sL_a', 'tI_a']
    relative_priors = {'mL': tss, 'mT': tts, 'mL_a': tss}
    shifted_priors = copy.deepcopy(priors)

    for pname, pval in shifted_priors.items():

        if pname in relative_priors.keys():

            if pval != None:

                ref = relative_priors[pname]
                distribution = pval['dist']

                if distribution== 'uniform':
                    pval['lower'] += ref
                    pval['upper'] += ref

                elif distribution == 'normal':
                    pval['mu'] += ref

                elif distribution == 'exponential':
                    pval['offset'] += ref

                elif distribution == 'gamma':
                    pval['offset'] += ref

                elif distribution == 'wald':
                    pval['alpha'] += ref

                elif distribution == 'constant':
                    pval['const'] += ref

                else:
                    raise ValueError(f'{distribution} not valid distribution.')
            
    return shifted_priors



def prior_config_old(priors, tss, tts):
    '''
    This function formats the priors dict generated by config_loader() so that 
    it conforms to the correct format to be loaded into a LIET model instance. 
    The input 'priors' is still in string format and must be evaluated for any 
    annotation offsets then converted to float/int. The two reference points 
    must be range shifted if RANGE_SHIFT=True in config
    
    NOTE: This is a silly function, but I'm currently keeping it separate in 
    the event that I change the way the config/initialization process is done.
    '''
    #absolute_priors = ['sL', 'tI', 'sT', 'w', 'sL_a', 'tI_a']
    relative_priors = {'mL': tss, 'mT': tts, 'mL_a': tss}
    shifted_priors = {}

    for pname, pval in priors.items():
        
        new_val = copy.deepcopy(pval)

        if pname in relative_priors.keys():

            if pval != None:

                ref = relative_priors[pname]

                if new_val[0] == 'uniform':
                    new_val[1] = [ref + e for e in new_val[1]]
                elif new_val[0] == 'normal':
                    new_val[1][0] += ref
                elif new_val[0] == 'exponential':
                    new_val[1][1] += ref
                elif new_val[0] == 'gamma':
                    new_val[1][2] += ref
                elif new_val[0] == 'wald':
                    new_val[1][2] += ref
                elif new_val[0] == 'constant':
                    new_val[1][0] += ref            

        shifted_priors[pname] = new_val
            
    return shifted_priors


## BEDGRAPH HANDLING ##########################################################

def gene_data(bedgraph, gene, pad_frac):
    '''
    Reads in BedGraph file for specified annotation ('gene' <dict>) and pads 
    the coverage data by a fraction ('pad_frac') of the overall annotation 
    length. Returns a numpy array for the genomic coordinates and a sorted 
    numpy array of integers for the read counts
    '''
    
    gchrom = gene['chrom']
    gstart = gene['start']
    gstop = gene['stop']
    gstrand = gene['strand']
    
    pad_len = int(pad_frac * (gstop - gstart))
    begin = gstart - pad_len
    end = gstop + pad_len
    
    xvals = np.array(range(begin, end))
    
    # Load in data
    data = []
    with open(bedgraph, 'r') as f:
    
        for line in f:
            line = line.strip().split()
        
            chm = str(line[0])
            stt = int(line[1])
            stp = int(line[2])
            cnt = int(line[3]) * gstrand
            
            if chm != gchrom or cnt <= 0:
                continue
            elif stt >= begin and stp <= end:
                data.extend(list(range(stt, stp))*cnt)
            else:
                continue
    
    data = sorted(data)
    data = np.array(data)
    return xvals, data



def rng_shift(gene, xvals, data):
    '''
    Adjusts the gene coordinates ('gene' <dict>) so that they are oriented 
    about zero and the data is oriented left to right for both strand. Returns 
    shifted xvals and corresponding shifted data. Fitting the model to this to 
    help aid in standardizing the parameter bounds.
    '''

    if gene['strand'] == 1:
        xvals = np.array(xvals) - gene['start']
        data = np.array(data) - gene['start']
    elif gene['strand'] == -1:
        xvals = (np.array(xvals) - gene['stop']) * (-1)
        xvals = np.flip(xvals, axis=0)
        data = (np.array(data) - gene['stop']) * (-1)
        data = np.flip(data, axis=0)
    else:
        raise ValueError("Must specify +1 or -1 for strand.")
        
    return xvals, data



## NEW STUFF ##################################################################

def overlap_check(annotations, pad=0):
    '''
    This function checks whether or not three sequential annotations are 
    overlapping for a given padding fraction. Returns 'True' or 'False'. The
    'pad' kwarg is the number of bases to add on to either side of the middle 
    annotation---i.e. (start-pad, stop+pad)
    '''
    l = annotations[0]
    m = annotations[1]
    r = annotations[2]
    
    lcomp = (l['start'] - m['stop'] - pad) * (l['stop'] - m['start'] + pad)
    rcomp = (r['start'] - m['stop'] - pad) * (r['stop'] - m['start'] + pad)
    
    # Check left
    if l['chrom'] != m['chrom'] or l['strand'] != m['strand'] or lcomp >= 0:
        l_overlap = False
    else:
        l_overlap = True

    if r['chrom'] != m['chrom'] or r['strand'] != m['strand'] or rcomp >= 0:
        l_overlap = False
    else:
        r_overlap = True

    if l_overlap or r_overlap:
        return True
    else:
        return False



def bg_iterator(bgfile, begin, end):
    '''
    This function takes a bedgraph file generator object and iterates through 
    it until the line that overlaps the 'begin' annotation is arrived at. 
    Returns the bedgraph line as a list, unless it's downstream of annot.
    '''
    for line in bgfile:
        line = line.strip().split('\t')
        ch = str(line[0])
        st = int(line[1])
        sp = int(line[2])
        ct = int(line[3])

        if sp <= begin:
            continue
        elif st <= end:
            return [ch, st, sp, ct]
        else:
            return None



def bgreads(bgfile, begin, end):
    '''
    bgfile      file object for bedgraph formated file (generator)
    begin       first coordinate (inclusive) to count reads (int)
    end         final coordinate (non-inclusive) to count reads (int)

    Returns: list of final bgfile line, pos-strand dict, neg-strand dict
    '''
    # Iterate through bgfile generator until it reaches 'begin' position
    for line in bgfile:
        line = line.strip().split('\t')
        st = int(line[1])
        sp = int(line[2])

        if sp <= begin:
            continue
        elif st <= end:
            ch = str(line[0])
            ct = int(line[3])
            line = [ch, st, sp, ct]
            break
        else:
            return None

    # Process first line and initialize the read strand dicts
    ch, st, sp, ct = line
    if ct >= 0:
        preads = {i:ct for i in range(max([begin, st]), sp)}
        nreads = {}
    else:
        preads = {}
        nreads = {i:abs(ct) for i in range(max([begin, st]), sp)}

    # Process all bg lines internal to annotation
    for line in bgfile:
        line = line.strip().split('\t')
        ch = str(line[0])
        st = int(line[1])
        sp = int(line[2])
        ct = int(line[3])
        
        if st < end or sp <= end:
            if ct >= 0:
                for i in range(max([begin, st]), min([end, sp])):
                    preads[i] = ct
            else:
                for i in range(max([begin, st]), min([end, sp])):
                    nreads[i] = abs(ct)
        else:
            return [ch, st, sp, ct], preads, nreads
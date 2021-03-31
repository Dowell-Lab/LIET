import numpy as np
import copy
from collections import defaultdict

# GENE ANNOTATION HANDLING ====================================================

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



# CONFIG FILE HANDLING ========================================================

def config_loader(config_file):
    '''
    Loads input parameters for model fitting. This includes model scope, 
    prior, data processing, fitting, and results formatting parameters.
    '''

    def float_int_cast(val):
        if '.' in val or 'e' in val:
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
        'FILES': {'ANNOTATION':'', 'BEDGRAPH':'', 'RESULTS':''},
        'MODEL': {'ANTISENSE':None, 'BACKGROUND':None, 'FRACPRIORS':None}, 
        'PRIORS': {'mL':None, 'sL':None, 'tI':None, 'mT':None, 'sT':None,
            'mL_a':None, 'sL_a':None, 'tI_a':None, 'w':None}, 
        'DATA_PROC': {'RANGE_SHIFT':None, 'PAD':None}, 
        'FIT': {'ITERATIONS':None, 'LEARNING_RATE':None, 'METHOD':None,
            'OPTIMIZER':None, 'MEANFIELD':None, 'TOLERANCE':None}, 
        'RESULTS': {'SAMPLES':None, 'MEAN':None, 'MODE':None, 'MEDIAN': None, 
            'STDEV':None, 'SKEW':None}
    }

    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line[0] == '#':
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
            # PRIORS
            if pname in prior_names:
                #pval = [e.strip() for e in pval.strip().split(',')]
                #pval = [pval[0], [float_int_cast(e) for e in pval[1:]]]
                #config[category][pname] = pval

                pval = [e.strip() for e in pval.strip().split(',')]         #### NEED TO DO SOMETHING HERE TO HAND IF pval IS None
                
                pval_dict = {}
                for p in pval:
                    var_name, var_val = p.strip().split(':')
                    if var_name == 'dist':
                        pval_dict[var_name] = var_val
                    else:
                        pval_dict[var_name] = float_int_cast(var_val)
                config[category][pname] = pval_dict
            # FILES
            elif (pname == 'ANNOTATION' or 
                pname == 'BEDGRAPH' or
                pname == 'RESULTS'):
                config[category][pname] = pval
            # MODEL
            elif (pname == 'ANTISENSE' or 
                pname == 'BACKGROUND' or
                pname == 'FRACPRIORS'):
                config[category][pname] = bool_cast(pval)
            # DATA_PROC
            elif pname == 'RANGE_SHIFT':
                config[category][pname] = bool_cast(pval)
            elif pname == 'PAD':
                pval = pval.strip().split(',')
                pval = [float_int_cast(i) for i in pval]
                config[category][pname] = pval
            # FIT
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
            elif (pname == 'METHOD' or
                pname == 'OPTIMIZER'):
                config[category][pname] = pval
            elif pname == 'MEANFIELD':
                config[category][pname] = bool_cast(pval)
            elif pname == 'TOLERANCE':
                config[category][pname] = float_int_cast(pval)
            # RESULTS
            elif pname == 'SAMPLES':
                pval = float_int_cast(pval)
                if isinstance(pval, int):
                    config[category][pname] = pval
                else:
                    raise ValueError(f"Input {pname} must be an integer.")
            elif pname in ['MEAN', 'MODE', 'MEDIAN', 'STDEV', 'SKEW']:
                config[category][pname] = bool_cast(pval)

            else:
                raise ValueError(f"Incorrect input parameter: {pname}")
    
    return config


# PRIORS HANDLING =============================================================

def prior_config(priors, tss, tcs, frac_priors=False):
    '''
    This function formats the priors dict generated by config_loader() so that 
    it conforms to the correct format to be loaded into a LIET model instance. 
    The input 'priors' is still in string format and must be evaluated for any 
    annotation offsets then converted to float/int. The two reference points 
    must be range shifted if RANGE_SHIFT=True in config
    
    NOTE: This is a silly function, but I'm currently keeping it separate in 
    the event that I change the way the config/initialization process is done.

    Parameters
    ----------
    tss : int
        Transcription start site

    tcs : int
        Transcription cleavage site
    
    Returns
    -------
    shifted_priors : dict
        Shifted priors dictionary, formatted for intput into LIET class 
        .set_priors() method. 
    '''
    #absolute_priors = ['sL', 'tI', 'sT', 'w', 'sL_a', 'tI_a']
    relative_priors = {'mL': tss, 'mT': tcs, 'mL_a': tss}
    len_scale = abs(tcs - tss)

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
                    if frac_priors:
                        pval['sigma'] *= len_scale

                elif distribution == 'exponential':
                    pval['offset'] += ref
                    if frac_priors:
                        pval['tau'] *= len_scale

                elif distribution == 'gamma':
                    pval['offset'] += ref
                    if frac_priors:
                        pval['sigma'] *= len_scale

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


# BEDGRAPH HANDLING ===========================================================

def bgline_cast(bgline):
    '''Casts the bgline (str) as a list with proper data types'''
    bglist = bgline.strip().split("\t")
    bglist[1] = int(bglist[1])
    bglist[2] = int(bglist[2])
    bglist[3] = int(bglist[3])
    return bglist


def bg2d(bglist):
    '''Convert bedgraph line <bglist> (len-4 list) to a read dict'''
    start = int(bglist[1])
    stop = int(bglist[2])
    count = int(bglist[3])
    
    if count >= 0:
        preads = {i:count for i in range(start, stop)}
        nreads = {}
        return preads, nreads
    else:
        preads = {}
        nreads = {i:-count for i in range(start, stop)}
        return preads, nreads

    
def bg2l(bglist):
    '''Convert bedgraph line <bglist> (len-4 list) to a read list'''
    start = int(bglist[1])
    stop = int(bglist[2])
    count = int(bglist[3])
    
    if count >= 0:
        preads = []
        nreads = []
        for i in range(start, stop):
            preads.extend([i]*count)
        return preads, nreads
    else:
        preads = []
        nreads = []
        for i in range(start, stop):
            nreads.extend([i]*-count)
        return preads, nreads

    
def reads_d2l(readdict):
    '''Convert bedgraph read dict to read list'''
    readlist = []
    for x, count in readdict.items():
        readlist.extend([x]*count)
        
    return readlist


def bglist_check(bglist, chromosome, begin, end):
    '''
    Checks if bglist is upstream, overlapping or downstream of annot. Returns 
    +1, 0, -1 respectively. Annotation given by <chromosome>, <begin>, and 
    <end>.
    '''
    chr_order = {
        'chr1':1, 'chr2':2, 'chr3':3, 'chr4':4, 'chr5':5, 'chr6':6, 'chr7':7, 
        'chr8':8, 'chr9':9, 'chr10':10, 'chr11':11, 'chr12':12, 'chr13':13, 
        'chr14':14, 'chr15':15, 'chr16':16, 'chr17':17, 'chr18':18, 'chr19':19,
        'chr20':20, 'chr21':21, 'chr22':22, 'chr23':23, 'chrX':24, 'chrY':25
    }

    chl = chr_order[bglist[0]]
    cha = chr_order[chromosome]
    i = bglist[1]
    f = bglist[2]
    
    # bedgraph list on upstream chromosome or upstream on same chromosome
    if chl < cha or (chl == cha and f < begin):
        return 1
    # bedgraph list on downstream chromosome or downstream on same chromosome
    elif chl > cha or (chl == cha and i > end):
        return -1
    # bedgraph list is overlapping annotation
    else:
        return 0


def add_bgd(bglist, begin, end, preads, nreads):
    '''
    Adds reads from <bglist> to read dictionaries <preads> and <nreads> for 
    annotation ranging from <begin> to <end>.
    '''
    ch, i, f, ct = bglist
    
    p, n = bg2d([ch, max(i, begin), min(f, end), ct])
    
    preads.update(p)
    nreads.update(n)


def add_bgl(bglist, begin, end, preads, nreads):
    '''
    Adds reads from <bglist> to read lists <preads> and <nreads> for 
    annotation ranging from <begin> to <end>.
    '''
    ch, i, f, ct = bglist
    
    p, n = bg2l([ch, max(i, begin), min(f, end), ct])
    
    preads.extend(p)
    nreads.extend(n)


def bgreads(bg, current_bgl, chromosome, begin, end):
    '''
    Primary method for processing bedgraph lines and converting them to read
    lists appropriate for loading into LIET class instance. 

    Parameters
    ----------
    bg : file object
        Bedgraph file object containing read count data. Must be in standard 
        four-column format (chr start stop count) and be sorted.
    
    current_bgl : list
        List containing information from the most recent bedgraph line, prior 
        to calling this function. Elements must be cast to appropriate data 
        type. Format: [chr, start, stop, count]

    chromosome : string
        String specifying the chromosome of the annotation being evaluated. 
        Must be in standard format: 'chr#'
    
    begin : int
        First genomic coordinate of the annotation being evaluated.

    end : int
        Last genomic coordinate of the annotation being evaluated.


    Returns
    -------
    bglist : list
        List containing information from most recent bedgraph line. The first 
        one downstream of annotation. Same format as <current_bgl>

    preads : dict
        Counter style dict containing the read counts on the positive strand 
        between genomic coordinates <begin> and <end>

    nreads : dict
        Counter style dict containing the read counts on the negative strand 
        between genomic coordinates <begin> and <end>
    '''
    preads = {}
    nreads = {}
    
    # Process current line
    loc = bglist_check(current_bgl, chromosome, begin, end)
    # Overlapping
    if loc == 0:
        add_bgd(current_bgl, begin, end, preads, nreads)
    # Downstream
    elif loc == -1:
        return current_bgl, preads, nreads
    # Upstream
    else:
        pass
    
    # Iterate through bedgraph until reaching first downstream line
    for bgline in bg:
        bglist = bgline_cast(bgline)
        loc = bglist_check(bglist, chromosome, begin, end)
        # Upstream
        if loc == 1:
            continue
        # Overlap
        elif loc == 0:
            add_bgd(bglist, begin, end, preads, nreads)
        # Downstream
        elif loc == -1:
            return bglist, preads, nreads
    # Return same thing if hits end of file
    else:
        return bglist, preads, nreads


def pad_calc(start, stop, strand, pad):
    '''
    Adjusts the (start, stop) interval to include the 5' and 3' padding.
    
    Parameters
    ----------
    start : int
        Starting genomic coordinate for loci (upstream of <stop> 
        regardless of strand)

    stop : int
        Final genomic coordinate for loci

    strand : str or int
        Strand on which loci is located
    
    pad : tuple
        Tuple containing 5' and 3' padding amounts (absolute or fraction). If 
        fractional, it's as a proportion of the loci length. Example, 
        '(200, 400)' would be 200bp upstream up 5' and 400bp down stream of 3'.
        Alternatively, '(0.1, 0.2)' for stop-start=100bp loci corresponds to
        '(10, 20)' in absolute numbers (round to nearest integer).


    Returns
    -------
    adj_start : int
        Start of adjusted genomic coordinate range

    adj_stop : int
        End of adjusted genomic coordinate range
    '''
    # Different pads for 5' and 3' end
    if isinstance(pad, (list, tuple)):
        pad5, pad3 = pad

        if isinstance(pad5, float):
            pad5 = int(abs(stop - start) * pad5)
        if isinstance(pad3, float):
            pad3 = int(abs(stop-start) * pad3)

        if strand == 1 or strand == "+":
            return start - pad5, stop + pad3
        else:
            return start - pad3, stop + pad5

    else:
        raise ValueError("pad must be either a list or tuple (5' pad, 3' pad")


#==============================================================================

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



# UNSORTED STUFF ==============================================================

def overlap_check2(begin1, end1, begin2, end2):
    '''Checks if two intervals are overlapping'''
    overlap = (end1 - begin2) * (end2 - begin1)
    if overlap > 0:
        return True
    else:
        return False


def overlap_check3(annotations, pad=0):
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



def bgreads_old(bgfile, begin, end):
    '''
    Parameters
    ----------
    bgfile : generator
        File object for bedgraph formated file (generator). This file must be
        sorted. The bedgraph file should contain 5'-end data.
    
    begin : int
        First coordinate (inclusive) to count reads
    
    end : int
        Final coordinate (non-inclusive) to count reads


    Returns
    -------
    current_line : list
        List of length 4 --- [chrom, start, stop, count]. Contains the column 
        values for the most recent bedgraph line from <bgfile>, once the <end> 
        coordinate has been reached.

    preads : list
        List of integers which correspond to the 5'-end coordiantes of all 
        positive strand reads, between coordinates <begin> and <end>.

    nreads : list
        List of integers which correspond to the 5'-end coordiantes of all 
        negative strand reads, between coordinates <begin> and <end>.
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
            current_line = [ch, st, sp, ct]
            return current_line, preads, nreads
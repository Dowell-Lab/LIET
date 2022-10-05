import sys
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


def annot_BED6_loader(annot_file, pad5, pad3):
    '''
    Loads annotations from BED6 file into a dictionary and then sorts 
    them by genomic coordinate and strand. Returns sorted dict of form:
    annot_sorted: {'chr#': {(start, stop, strand): gene_id, ...}, ...}
    pad: {gene_id: (pad5, pad3), ... }
    
    Only uses the following fields from the BED6 formatted input: 
    chr (0), start (1), stop (2), name (3), strand (5)
    '''
    strand_dict = {'+': 1, '-': -1}
    pad = {}

    # Initialize annotation dictionary with chrom ID's in annot file
    with open(annot_file, 'r') as af:
        chromosomes = set()
        for line in af:
            chrom = str(line.strip().split('\t')[0])
            chromosomes.add(chrom)
    
    annot = {ch:{} for ch in sorted(chromosomes)}

    # Open BED6 file (tab delimited: chr, start, stop, id, score, strand)
    with open(annot_file, 'r') as af:

        for i, line in enumerate(af):
            line = line.strip().split('\t')

            if len(line) != 6:
                raise ValueError(f"Annotation at line {i} is incorrectly "
                    f"formatted. See: '{line}'")

            try:
                chrom = str(line[0])
                start = int(line[1])
                stop = int(line[2])
                id = str(line[3])
                strnd = strand_dict[line[5]]
                annot[chrom][(start, stop, strnd)] = id

                pad[id] = (int(pad5), int(pad3))

            except KeyError:
                raise KeyError(f"One or both keys {chrom} and {id} do not"
                    "exist.")
            except:
                raise ValueError(f"Can't parse line {i}. Incorrectly "
                    f"formatted. See: '{line}'")

    # Sort the regions of interest for each chromosome and write to out dict
    annot_sorted = {}
    for ch, rois in annot.items():
        if rois != {}:
            annot_sorted[ch] = {r:annot[ch][r] for r in sorted(rois.keys())}
        else:
            continue
    
    return annot_sorted, pad


def annot_loader(annot_file):
    '''
    NOTE: Depricated. Replaced by annot_BED6_loader()

    Loads annotations from basic gtf file into a dictionary and then sorts 
    them by genomic coordinate and strand. Returns sorted dict of form:
    annot_sorted: {'chr#': {(start, stop, strand): gene_id, ...}, ...}
    pad: {gene_id: (pad5, pad3), ... }
    '''

    chromosomes = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chr23', 'chrX', 'chrY'
    ]
    annot = {ch:{} for ch in chromosomes}
    pad = {}
    strand_dict = {'+': 1, '-': -1}

    # Open annotation file (tab delimited: chr, start, stop, strand, id)
    with open(annot_file, 'r') as f:

        for i, line in enumerate(f):
            line = line.strip().split('\t')

            if len(line) != 6:
                raise ValueError(f"Annotation at line {i} is incorrectly "
                    f"formatted. See: '{line}'")

            try:
                chrom = str(line[0])
                start = int(line[1])
                stop = int(line[2])
                strnd = strand_dict[line[3]]
                id = str(line[4])
                annot[chrom][(start, stop, strnd)] = id

                pad_vals = line[5].strip().split(',')
                pad5 = int(pad_vals[0])
                pad3 = int(pad_vals[1])
                pad[id] = (pad5, pad3)

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

    return annot_sorted, pad



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
        'DATA_PROC': {'RANGE_SHIFT':None, 'PAD':None, 'COV_THRESHOLDS':None}, 
        'FIT': {'ITERATIONS':None, 'LEARNING_RATE':None, 'METHOD':None,
            'OPTIMIZER':None, 'MEANFIELD':None, 'TOLERANCE':None}, 
        'RESULTS': {'SAMPLES':None, 'MEAN':None, 'MODE':None, 'MEDIAN': None, 
            'STDEV':None, 'SKEW':None, 'PDF': False}
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
                pname == 'BEDGRAPH_POS' or
                pname == 'BEDGRAPH_NEG' or
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
            elif (pname == 'PAD' or
                pname == 'COV_THRESHOLDS'):
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
            elif pname == 'PDF':
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



# def prior_config_old(priors, tss, tts):
#     '''
#     This function formats the priors dict generated by config_loader() so that 
#     it conforms to the correct format to be loaded into a LIET model instance. 
#     The input 'priors' is still in string format and must be evaluated for any 
#     annotation offsets then converted to float/int. The two reference points 
#     must be range shifted if RANGE_SHIFT=True in config
    
#     NOTE: This is a silly function, but I'm currently keeping it separate in 
#     the event that I change the way the config/initialization process is done.
#     '''
#     #absolute_priors = ['sL', 'tI', 'sT', 'w', 'sL_a', 'tI_a']
#     relative_priors = {'mL': tss, 'mT': tts, 'mL_a': tss}
#     shifted_priors = {}

#     for pname, pval in priors.items():
        
#         new_val = copy.deepcopy(pval)

#         if pname in relative_priors.keys():

#             if pval != None:

#                 ref = relative_priors[pname]

#                 if new_val[0] == 'uniform':
#                     new_val[1] = [ref + e for e in new_val[1]]
#                 elif new_val[0] == 'normal':
#                     new_val[1][0] += ref
#                 elif new_val[0] == 'exponential':
#                     new_val[1][1] += ref
#                 elif new_val[0] == 'gamma':
#                     new_val[1][2] += ref
#                 elif new_val[0] == 'wald':
#                     new_val[1][2] += ref
#                 elif new_val[0] == 'constant':
#                     new_val[1][0] += ref            

#         shifted_priors[pname] = new_val
            
#     return shifted_priors


# BEDGRAPH HANDLING ===========================================================

def bgline_cast(bgline):
    '''Casts the bgline (str) as a list with proper data types'''
    bglist = bgline.strip().split("\t")
    bglist[1] = int(bglist[1])
    bglist[2] = int(bglist[2])
    bglist[3] = int(bglist[3])
    return bglist


def bg2d(bglist):
    '''Convert bedgraph list <bglist> (len-4 list) to a read dict'''
    start = bglist[1]
    stop = bglist[2]
    count = abs(bglist[3])
    
    if count > 0:
        reads = {i:count for i in range(start, stop)}
        return reads
    else:
        return {}

# def bg2d(bglist): 
#     '''Convert bedgraph line <bglist> (len-4 list) to a read dict'''
#     start = int(bglist[1])
#     stop = int(bglist[2])
#     count = int(bglist[3])
    
#     if count >= 0:
#         preads = {i:count for i in range(start, stop)}
#         nreads = {}
#         return preads, nreads
#     else:
#         preads = {}
#         nreads = {i:-count for i in range(start, stop)}
#         return preads, nreads


# def bg2l(bglist):
#     '''Convert bedgraph line <bglist> (len-4 list) to a read list'''
#     start = int(bglist[1])
#     stop = int(bglist[2])
#     count = int(bglist[3])
    
#     if count > 0:
#         preads = []
#         nreads = []
#         for i in range(start, stop):
#             preads.extend([i]*count)
#         return preads, nreads
#     else:
#         preads = []
#         nreads = []
#         for i in range(start, stop):
#             nreads.extend([i]*-count)
#         return preads, nreads


def reads_d2l(readdict):
    '''Convert bedgraph read dict to read list'''
    readlist = []
    for x, count in readdict.items():
        readlist.extend([x]*count)
        
    return readlist


def chrom_order_reader(bedgraph_file1, bedgraph_file2):
    '''
    Create a dictionary containing an ordered set of chromosome strings. The 
    chromosome strings are the keys and the values are the indexing values. 
    The strings are sourced from the two bedgraph files, preserving the order. 
    Format: {'chr1': 1, 'chr2': 2, ...}
    '''
    with open(bedgraph_file1, 'r') as bgf:
        chrom1 = [line.strip().split('\t')[0] for line in bgf]
    with open(bedgraph_file2, 'r') as bgf:
        chrom2 = [line.strip().split('\t')[0] for line in bgf]

    chrom1 = sorted(set(chrom1), key=chrom1.index)
    chrom2 = sorted(set(chrom2), key=chrom2.index)

    common = set(chrom1).intersection(set(chrom2))

    if set(chrom1) != common:
        print("WARNING: BedGraph files do not have all the same chromosomes.", 
            file=sys.stderr)
    
    chr_order = dict([tuple(reversed(t)) for t in enumerate(chrom1) if t[1] in common])
    return chr_order
    

def bglist_check(bglist, chromosome, begin, end, chr_order):
    '''
    Checks if <bglist> is upstream, overlapping or downstream of annot. Returns
     +1, 0, -1 respectively. Annotation given by <chromosome>, <begin>, and 
    <end>. Bedgraph line <bglist> is of the form [chr_id, initial, final, ...].
    This function will return None if the bedgraph line chromsome ID or the 
    annotation chromosome ID are not contained in the <chr_order> reference 
    dictionary which contains the chromosome ordering.
    '''
    #chr_order = {
    #    'chr1':1, 'chr2':2, 'chr3':3, 'chr4':4, 'chr5':5, 'chr6':6, 'chr7':7, 
    #    'chr8':8, 'chr9':9, 'chr10':10, 'chr11':11, 'chr12':12, 'chr13':13, 
    #    'chr14':14, 'chr15':15, 'chr16':16, 'chr17':17, 'chr18':18, 'chr19':19,
    #    'chr20':20, 'chr21':21, 'chr22':22, 'chr23':23, 'chrX':24, 'chrY':25
    #}

    # Determine bedgraph line and annot chromosome indexes
    try:
        bg_chrom = bglist[0]
        bg_idx = chr_order[bg_chrom]
    except KeyError:
        print(f"WARNING: {bg_chrom} is not contained in shared chr IDs",
            file=sys.stderr)
        return None
    try:
        annot_idx = chr_order[chromosome]
    except KeyError:
        print(f"WARNING: {chromosome} is not contained in shared chr IDs", 
            file=sys.stderr)
        return None

    # Initial/final base positions for bedGraph line
    initial = bglist[1]
    final = bglist[2]
    
    # bedgraph list on upstream chromosome or upstream on same chromosome
    if bg_idx < annot_idx or (bg_idx == annot_idx and final < begin):
        return 1
    # bedgraph list on downstream chromosome or downstream on same chromosome
    elif bg_idx > annot_idx or (bg_idx == annot_idx and initial > end):
        return -1
    # bedgraph list is overlapping annotation
    else:
        return 0


def add_bg_dict(bglist, begin, end, reads_dict):
    '''
    Adds reads from <bglist> to correct read dictionary <preads> and <nreads> 
    for annotation ranging from <begin> to <end>.
    '''
    ch, i, f, ct = bglist
    
    overlap_region = [ch, max(i, begin), min(f, end), ct]
    reads = bg2d(overlap_region)
    
    if ct > 0:
        reads_dict.update(reads)


# def add_bg_dict(bglist, begin, end, preads, nreads):
#     '''
#     Adds reads from <bglist> to correct read dictionary <preads> and <nreads> 
#     for annotation ranging from <begin> to <end>.
#     '''
#     ch, i, f, ct = bglist
    
#     reads = bg2d([ch, max(i, begin), min(f, end), ct])
    
#     if ct > 0:
#         preads.update(reads)
#     elif ct < 0:
#         nreads.update(reads)


# def add_bgl(bglist, begin, end, preads, nreads):
#     '''
#     Adds reads from <bglist> to read lists <preads> and <nreads> for 
#     annotation ranging from <begin> to <end>.
#     '''
#     ch, i, f, ct = bglist
    
#     p, n = bg2l([ch, max(i, begin), min(f, end), ct])
    
#     preads.extend(p)
#     nreads.extend(n)


# def bgreads(bg, current_bgl, chromosome, begin, end):
#     '''
#     Primary method for processing bedgraph lines and converting them to read
#     lists appropriate for loading into LIET class instance. 

#     Parameters
#     ----------
#     bg : file object
#         Bedgraph file object containing read count data. Must be in standard 
#         four-column format (chr start stop count) and be sorted.
    
#     current_bgl : list
#         List containing information from the most recent bedgraph line, prior 
#         to calling this function. Elements must be cast to appropriate data 
#         type. Format: [chr, start, stop, count]

#     chromosome : string
#         String specifying the chromosome of the annotation being evaluated. 
#         Must be in standard format: 'chr#'
    
#     begin : int
#         First genomic coordinate of the annotation being evaluated.

#     end : int
#         Last genomic coordinate of the annotation being evaluated.


#     Returns
#     -------
#     bglist : list
#         List containing information from most recent bedgraph line. The first 
#         one downstream of annotation. Same format as <current_bgl>

#     preads : dict
#         Counter style dict containing the read counts on the positive strand 
#         between genomic coordinates <begin> and <end>

#     nreads : dict
#         Counter style dict containing the read counts on the negative strand 
#         between genomic coordinates <begin> and <end>
#     '''
#     preads = {}
#     nreads = {}
    
#     # Process current line
#     loc = bglist_check(current_bgl, chromosome, begin, end)
#     # Overlapping
#     if loc == 0:
#         add_bg_dict(current_bgl, begin, end, preads, nreads)
#     # Downstream
#     elif loc == -1:
#         return current_bgl, preads, nreads
#     # Upstream
#     else:
#         pass
    
#     # Iterate through bedgraph until reaching first downstream line
#     for bgline in bg:
#         bglist = bgline_cast(bgline)
#         loc = bglist_check(bglist, chromosome, begin, end)
#         # Upstream
#         if loc == 1:
#             continue
#         # Overlap
#         elif loc == 0:
#             add_bg_dict(bglist, begin, end, preads, nreads)
#         # Downstream
#         elif loc == -1:
#             return bglist, preads, nreads
#     # Return same thing if hits end of file
#     else:
#         return bglist, preads, nreads


def bgreads(bg, current_bg_list, chromosome, begin, end, chr_order):
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
        Must be in standard hg38 format: 'chr#'
    
    begin : int
        First genomic coordinate of the annotation being evaluated.

    end : int
        Last genomic coordinate of the annotation being evaluated.

    chr_order : dict
        Dictionary containing the ordering of chromosome strings. The keys are 
        the chromosome ID strings read from the bedgraph files and the values 
        are their numeric indexes.

    Returns
    -------
    bglist : list
        List containing information from most recent bedgraph line. The first 
        one downstream of annotation. Same format as <current_bgl>

    reads : dict
        `Counter` style dict containing the read counts on one of the strands 
        between genomic coordinates <begin> and <end>
    '''
    # Dict for reads on one of the strands
    reads = {}
    
    # Process current line
    loc = bglist_check(current_bg_list, chromosome, begin, end, chr_order)
    # Overlapping
    if loc == 0:
        add_bg_dict(current_bg_list, begin, end, reads) #Update w/ new reads
    # Downstream
    elif loc == -1:
        return current_bg_list, reads
    # Upstream (1) or wrong chromosome (None)
    else:
        pass
    
    # Iterate through bedgraph until reaching first downstream line
    for bgline in bg:
        bglist = bgline_cast(bgline)
        loc = bglist_check(bglist, chromosome, begin, end, chr_order)
        # Upstream or invalid chromosome
        if loc == 1 or loc == None:
            continue
        # Overlap
        elif loc == 0:
            add_bg_dict(bglist, begin, end, reads)  # Update w/ new reads
        # Downstream
        elif loc == -1:
            return bglist, reads
    # Return same thing if hits end of file
    else:
        return bglist, reads


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


def bedgraph_loader(bgp_file, bgn_file, annot_dict, pad_dict, chr_order=None):
    '''
    This is the main function for reading in the entirety of a pair of bedgraph
     files (pos and neg strands), for all the annotations in a annot dict. It 
    replaces the annot_dict loop in the old liet_exe.py.
    
    Parameters
    ----------
    bgp_file : str
        Path to positive strand bedgraph file.
    
    bgn_file : str
        Path to negative strand bedgraph file.

    annot_dict : dict
        Annotation dictionary containing gene IDs as well as their genomic 
        coordinates extracted from a GTF. Generated by annot_BED6_loader(). 
        Format: {'chr#': {(start, stop, strand): gene_id, ...}, ...}
        NOTE: annot_dict keys must be sorted by chromosome and coordinates

    pad_dict : dict
        Coordinate padding dictionary containing 5' and 3' padding lengths 
        (measured in # of basepair) for each gene ID in annot_dict. Generated 
        by annot_load() at the same time as annot_dict. Format: 
        {'gene_id': (5'pad, 3'pad), ...}

    order : dict
        External chromosome order. If None (default) order is generated by the 
        chrom_order_reader() function.

    Returns
    -------
    read_dict : dict
        Dictionary containing the reads from the bedgraph file for each ID in 
        the annot_dict. The reads are returned as a Counter style dictionary.
        Format: {'gene_id': (preads, nreads), ...}
    '''
    read_dict = {}

    # Determine chromosome string order if none provided
    if chr_order == None:
        order = chrom_order_reader(bgp_file, bgn_file)

    with open(bgp_file, 'r') as bgp, open(bgn_file, 'r') as bgn:
        current_bgpl = ['chr1', 0, 0, 0]    # Initialize pos strand bg line
        current_bgnl = ['chr1', 0, 0, 0]    # Initialize neg strand bg line
        
        ## MAIN LOOP OVER ANNOTATIONS =========================================
        for chromosome, annotations in annot_dict.items():

            for region, gene_id in annotations.items():                     # I SHOULD CHECK THIS IS SORTED
            
                start, stop, strand = region
                
                # Compute padded range
                pad = pad_dict[gene_id]
                pad_args = [start, stop, strand, pad]
                adj_start, adj_stop = pad_calc(*pad_args)

                # Extract reads
                bgp_args = [
                    bgp, 
                    current_bgpl, 
                    chromosome, 
                    adj_start, 
                    adj_stop,
                    chr_order
                ]
                current_bgpl, preads = bgreads(*bgp_args)
                
                bgn_args = [
                    bgn, 
                    current_bgnl, 
                    chromosome, 
                    adj_start, 
                    adj_stop,
                    chr_order
                ]
                current_bgnl, nreads = bgreads(*bgn_args)

                # Assign reads to output dict
                read_dict[gene_id] = (preads, nreads)

    return read_dict


# FILTERING ===================================================================

def cov_filter(reads, annotations, thresholds=(0, 0)):
    '''
    This funciton is used to filter out genes with low or no coverage on the 
    sense and/or anti-sense strands. The threshold values (sense, anti-sense) 
    are compared to values of reads dict and then removed from it as well as 
    the annotation dict.

    Parameters
    ----------
    reads : dict
        Dictionary containing read data from bedGraph files. Reads are in a
        Counter-like dictionary {<base position> : <count>, ...}
        Format: {'gene_id': (preads, nreads), ...}
    
    annots : dict
        Dictionary continaing annotations for each gene to be fit.
        Format: {'chr#': {(start, stop, strand): gene_id, ...}, ...}

    thresholds : tuple
        Sense and anti-sense strand coverage threshold for filter (# reads).

    Returns
    -------
    reads_filtered : dict
        Dictionary containing filtered reads, formatted same as input.

    annots_filtered : dict
        Dictionary containing filtered annotations, formatted same as input.

    err_info : list
        Information to be logged on which genes had been removed and why.
    '''

    # Extract strand info
    strands = {gene_id: annot[2] for chrom_annots in annotations.values() 
               for annot, gene_id in chrom_annots.items()}

    remove_genes = set()
    log_info = {}

    for gene, strand in strands.items():

        # Sense (map: strand +1 --> index 0 and -1 --> 1)
        sense_cov = sum(reads[gene][round((1-strand)/2)].values())
        # Anti-sense (map: strand +1 --> index 1 and -1 --> 0)
        antisense_cov = sum(reads[gene][round((1+strand)/2)].values())
    
        if sense_cov <= thresholds[0] and antisense_cov > thresholds[1]:
            remove_genes.add(gene)
            log_str = f"Cov filter ({sense_cov}, {antisense_cov}):sense\n"
            log_info[gene] = log_str

        elif sense_cov > thresholds[0] and antisense_cov <= thresholds[1]:
            remove_genes.add(gene)
            log_str = f"Cov filter ({sense_cov}, {antisense_cov}):anti\n"
            log_info[gene] = log_str

        elif sense_cov <= thresholds[0] and antisense_cov <= thresholds[1]:
            remove_genes.add(gene)
            log_str = f"Cov filter ({sense_cov}, {antisense_cov}):sense,anti\n"
            log_info[gene] = log_str

        else:
            continue

    reads_filtered = {g: r for g, r in reads.items() if g not in remove_genes}
    annots_filtered = {}
    for chromosome, chrom_annots in annotations.items():
        filt = {a: g for a, g in chrom_annots.items() if g not in remove_genes}
        annots_filtered[chromosome] = filt

    return reads_filtered, annots_filtered, log_info



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
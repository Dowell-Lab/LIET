import sys
from collections import OrderedDict

class FitParse:
    '''
    FitParse loads the results of a LIET fitting run from the standard output 
    file. Results are organzed into several dictionaries and lists for easier 
    parsing. Additionally, the class contains methods for parsing the main 
    results dictionary <fits>.
    '''

    def __init__(self, res_file, log_file=None):

        self.definitions = OrderedDict({
            "mL": "Sense strand loading position (mu)",
            "sL": "Sense strand loading stdev (sigma)",
            "tI": "Sense strand initiation length (tau)",
            "mT": "Sense strand termination position (mu) relative to TCS",
            "sT": "Sense strand termination stdev (sigma)",
            "w": "Sense strand weights [load, elong, terminate, background]",
            "mL_a": "Antisense strand loading position (mu)",
            "sL_a": "Antisense strand loading stdev (sigma)",
            "tI_a": "Antisense strand initiation length (tau)",
            "w_a": "Antisense strand weights [load, background]",
        })

        self.genes = []
        self.annotations = OrderedDict()
        self.fits = OrderedDict()

        with open(res_file, 'r') as rf:

            for line in rf:
                # Iterate through header
                if line[0] == '#':
                    line_list = line[1:].strip().split()
                    if line_list[0] == "CONFIG":
                        self.config = line_list[1]
                    continue
                else:
                    pass
                
                # Check line (gene) has a fit result
                line_list = line.strip().split('\t')
                if len(line_list) != 6:
                    if "error" in line_list:
                        print(line_list, file=sys.stderr)
                        continue
                    else:
                        print(f"CHECK LINE: {line_list}", file=sys.stderr)
                        continue
                
                # Parse and cast line
                chrom, start, stop, strand, gid, fit = line_list
                start = int(start)
                stop = int(stop)
                strand = int(strand)

                self.genes.append(gid)
                self.annotations[gid] = (chrom, start, stop, strand)

                # Fit line format: param1=val1:err1,param2=val2:err2,...
                # Parse all the best fit parameter values (CSV string)
                temp = OrderedDict()
                for param_val in fit.strip().split(','):
                    # Parameter name and its value
                    p, v = param_val.strip().split('=')
                    # Mean and standard error of value (from posterior dist)
                    v_m, v_s = v.split(':')
                    if p in ['w', 'w_a']:
                        v_m = [float(i) for i in v_m.strip('[]').split()]
                        v_s = [float(i) for i in v_s.strip('[]').split()]
                    else:
                        v_m = float(v_m)
                        v_s = float(v_s)
                    temp[p] = (v_m, v_s)
                
                self.fits[gid] = temp

        # Extract and assign all the variable arrays
        self.mL, self.mL_std = self.param_extract('mL', stdev=True)
        self.sL, self.sL_std = self.param_extract('sL', stdev=True)
        self.tI, self.tI_std = self.param_extract('tI', stdev=True)
        
        # Recalculate mT values so they are relative to end of annotation
        absolute_mT, self.mT_std = self.param_extract('mT', stdev=True)
        relative_mT = []
        for i, gene in enumerate(self.genes):
            tss = self.annotations[gene][1]
            tcs = self.annotations[gene][2]
            diff = abs(tcs - tss)
            relative_mT.append(absolute_mT[i] - diff)
        self.mT = relative_mT

        self.sT, self.sT_std = self.param_extract('sT', stdev=True)
        self.w, self.w_std = self.param_extract('w', stdev=True)
        self.mL_a, self.mL_a_std = self.param_extract('mL_a', stdev=True)
        self.sL_a, self.sL_a_std = self.param_extract('sL_a', stdev=True)
        self.tI_a, self.tI_a_std = self.param_extract('tI_a', stdev=True)
        self.w_a, self.w_a_std = self.param_extract('w_a', stdev=True)
    
        if log_file:
            self.log = OrderedDict()
            with open(log_file, 'r') as lf:
                for line in lf:
                    if line[0] == '#':
                        continue
                    elif line[0] == '>':
                        line = line.strip().split(':')
                        gene_id = line[0][1:]
                        self.log[gene_id] = dict()
                    else:
                        print(line)
                        field, value = line.strip().split(':')
                        if field == 'strand_cov':
                            value = value.strip('()').split(',')
                            value = tuple(map(int, value))
                        elif field == 'elbo_range':
                            value = value.strip('()').split(',')
                            value = tuple(map(float, value))
                        else:
                            continue
                        self.log[gene_id].update({field: value})



    def param_extract(self, p, stdev=False):
        '''
        Extract param (p) values (ordered based on genes list) from fits 
        dictionary and output them to a list.
        '''
        param_vals = []
        if stdev:
            param_stdev = []

        for g in self.genes:
            param_vals.append(self.fits[g][p][0])
            if stdev:
                param_stdev.append(self.fits[g][p][1])

        if stdev:
            return param_vals, param_stdev
        else:
            return param_vals


# Intersecting function for instances of FitParse class =======================
def fitparse_intersect(*samples, stdev=True):
    '''
    Identifies the gene fit results that are common to all class instances and 
    filters all fit parameter results to only include those from the common 
    set of genes. 
    '''
    # Identify genes shared across all fits
    gene_sets_list = [set(i.genes) for i in samples]
    gene_set_overlap = set.intersection(*gene_sets_list)
    intersect_genes = list(
        filter(lambda gene: gene in gene_set_overlap, samples[0].genes)
    )

    for samp in samples:
        # Identify original indexes for intersect genes
        gene_indexes = [samp.genes.index(g) for g in intersect_genes]
        # Reassign gene list (order preserved)
        samp.genes = intersect_genes
        
        for param in samp.definitions.keys():
            # Select param values based on intersected gene indexes and 
            # reassign to fit class
            intersect_param_vals = [
                val for i, val in enumerate(vars(samp)[param]) 
                if i in gene_indexes
            ]
            vars(samp).update({param: intersect_param_vals})
            
            if stdev:
                intersect_param_std = [
                    val for i, val in enumerate(vars(samp)[param+"_std"]) 
                    if i in gene_indexes
                ]
                vars(samp).update({param+"_std": intersect_param_std})

        # Filter the fits and annotations dict on intersect genes
        original_genes = list(samp.fits.keys())
        for gene in original_genes:
            if gene in intersect_genes:
                continue
            else:
                samp.fits.pop(gene)
                samp.annotations.pop(gene)
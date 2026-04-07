import sys
import pandas as pd
from collections import OrderedDict

class FitParse:
    '''
    FitParse loads the results of a LIET fitting run from the standard output 
    file. Results are organzed into several dictionaries and lists for easier 
    parsing. Additionally, the class contains methods for parsing the main 
    results dictionary and produces a dataframe accessible by results.df <fits>.
    '''

    def __init__(self, res_file, log_file=None, antisense=True, ET_sense=True, ET_antisense=False, colon_format=False, debug=False):

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
            "mT_a": "Antisense strand termination position (mu) relative to mu-TCS",
            "sT_a": "Antisense strand termination stdev (sigma)",
            "w_a": "Antisense strand weights [load, background]",
        })

        self.antisense = antisense
        self.ET_sense = ET_sense
        self.ET_antisense = ET_antisense
        self.genes = []
        self.annotations = OrderedDict()
        self.fits = OrderedDict()
        self.percentiles = None
        self.priors_list = ['mL','sL', 'tI', 'mT','sT','w_LI','w_E', 'w_T', 'w_B', 
                            'mL_a','sL_a', 'tI_a', 'mT_a', 'sT_a', 'w_aLI','w_aE', 'w_aT', 'w_aB']

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
                        if debug:
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
                    # param is Percentiles, need to change data stripping
                    if p == "Percentiles":
                        self.percentiles = True
                        temp["Percentiles"]= v
                    else:
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

        # Extract and assign variable arrays
        self.mL, self.mL_std = self.param_extract('mL', stdev=True)
        self.sL, self.sL_std = self.param_extract('sL', stdev=True)
        self.tI, self.tI_std = self.param_extract('tI', stdev=True)
        
        # Recalculate mT values so they are relative to end of annotation
        if ET_sense:
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
        
        if antisense is True:
            self.mL_a, self.mL_a_std = self.param_extract('mL_a', stdev=True)
            self.sL_a, self.sL_a_std = self.param_extract('sL_a', stdev=True)
            self.tI_a, self.tI_a_std = self.param_extract('tI_a', stdev=True)
            self.w_a, self.w_a_std = self.param_extract('w_a', stdev=True)
            if ET_antisense:
                absolute_mT_a, self.mT_a_std = self.param_extract('mT_a', stdev=True)
                relative_mT_a = []
                for i, gene in enumerate(self.genes):
                    tss = self.annotations[gene][1]
                    tcs = self.annotations[gene][2]
                    diff = abs(tcs - tss)
                    relative_mT_a.append(absolute_mT_a[i] - diff)
                self.mT_a = relative_mT_a
                self.sT_a, self.sT_a_std = self.param_extract('sT_a', stdev=True)
        
        ## make the unique weighted lists
#         self.priors_list = ['mL','sL', 'tI', 'mT','sT','w_LI','w_E', 'w_T', 'w_B', 'mL_a','sL_a', 'w_aLI','w_aB']
        if ET_sense:
            self.w_LI, self.w_E, self.w_T, self.w_B  = self.weight_par_extract(self.w)
            self.w_LI_std, self.w_E_std, self.w_T_std, self.w_B_std = self.weight_par_extract(self.w_std)
        else:
            self.w_LI, self.w_B  = self.weight_par_extract(self.w)
            self.w_LI_std, self.w_B_std = self.weight_par_extract(self.w_std)
        if antisense:
            if ET_antisense:
                self.w_aLI, self.w_aE, self.w_aT, self.w_aB  = self.weight_par_extract(self.w_a)
                self.w_aLI_std, self.w_aE_std, self.w_aT_std, self.w_aB_std = self.weight_par_extract(self.w_a_std)
            else:
                self.w_aLI, self.w_aB = self.weight_par_extract(self.w_a)
                self.w_aLI_std, self.w_aB_std = self.weight_par_extract(self.w_a_std)
        
        
        
        # Parse strand coverage and min/max elbo values from log file
        # colon format clarified if enhancer so chr:X in name
        if log_file:
            self.log = OrderedDict()
            with open(log_file, 'r') as lf:
                for line in lf:
                    if line[0] == '#':
                        continue
                    elif line[0] == '>':
                        line = line.strip().split(':')
                        if colon_format:
                            gene_id = ":".join([line[0][1:], line[1]])
                        else:
                            gene_id = line[0][1:]
                        self.log[gene_id] = dict()
                    else:
                        field, value = line.strip().split(':')
                        if field == 'fit_range':
                            value = value.strip('()').split(',')
                            value = tuple(map(int, value))
                        elif field == 'strand_cov':
                            value = value.strip('()').split(',')
                            value = tuple(map(int, value))
                        elif field == 'elbo_range':
                            value = value.strip('()').split(',')
                            value = tuple(map(float, value))
                        elif field == 'fit_time_min':
                            value = float(value)
                        else:
                            continue
                        self.log[gene_id].update({field: value})


            
            self.cov_pos = []
            self.cov_neg = []
            print("Number of features considered:", len(self.genes))
             for g in self.genes:
                 if (self.log[g]['strand_cov']):
                     self.cov_pos.append(self.log[g]['strand_cov'][0])
                     self.cov_neg.append(abs(self.log[g]['strand_cov'][1]))

        # Get all info in a dataframe
        self.df = self.dataframe_creator()

    
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
    
    def percentile_extract(self):
        """
        Extract the percentiles (ordered baased on genes list) from fits
        dictionary and output as a dictionary with the different percentiles as keys
        and the lists as the lists in gene order.
        """
        percentile_dict = dict()
        # save each percentile as per_pos and per_neg
        first = True
        for g in self.genes:
            if first:
                for perc in self.fits[g]["Percentiles"].split(";"):
                    p_new, pos_res, neg_res = perc.split(":")
                    percentile_dict["_".join([p_new, "perc_pos"])] = [int(pos_res)]
                    percentile_dict["_".join([p_new, "perc_neg"])] = [int(neg_res)]
                    first = False
            else:
                for perc in self.fits[g]["Percentiles"].split(";"):
                    p_new, pos_res, neg_res = perc.split(":")
                    percentile_dict["_".join([p_new, "perc_pos"])] = percentile_dict["_".join([p_new, "perc_pos"])] + [int(pos_res)]
                    percentile_dict["_".join([p_new, "perc_neg"])] = percentile_dict["_".join([p_new, "perc_neg"])] + [int(neg_res)]
        return percentile_dict
            
    def weight_par_extract(self, par_list):
        '''This function will take the par list and separate it, returning new lists of each of the indexes'''
        list1 = []
        list2 = []
        list3 = []
        list4 = []
        if len(par_list[0]) == 4:
            for line in par_list:
                list1 = list1 + [line[0]]
                list2 = list2 + [line[1]]
                list3 = list3 + [line[2]]
                list4 = list4 + [line[3]]
            return list1, list2, list3, list4
        elif len(par_list[0]) == 2:
            for line in par_list:
                list1 = list1 + [line[0]]
                list2 = list2 + [line[1]]
            return list1, list2
        else: print("There are not 2/4 indexes")      
        
    def dataframe_creator(self):
        "This function creates a dataframe where each gene is a column with labeled rows of all fit information"
        '''
            Columns: 
        Annotations: 'chrom', 'start', 'stop', 'strand' 
        Sense Main Priors: 'mL_mean', 'mL_stdev', 'sL_mean','sL_stdev', 'tI_mean', 
                           'tI_stdev', 'mT_mean', 'mT_stdev', 'sT_mean','sT_stdev' 
        Sense Weight Priors:'w_LI_mean', 'w_LI_stdev', 'w_E_mean', 'w_E_stdev',
                            'w_T_mean', 'w_T_stdev', 'w_B_mean', 'w_B_stdev' 
        Antisense Main Priors:'mL_a_mean','mL_a_stdev', 'sL_a_mean', 'sL_a_stdev', 'tI_a_mean', 'tI_a_stdev'
        Antisense Weight Priors:'w_aLI_mean', 'w_aLI_stdev','w_aB_mean', 'w_aB_stdev'
        Log Info: 'pos_cov', 'neg_cov', 'total_cov', 'elbo_lrange', 'elbo_urange', 'fit_time_min' 
        '''
        # Create a dataframe based on annotations
        self.df = pd.DataFrame(data=self.annotations, index=["chrom", "start", "stop", "strand"]).transpose()
        self.df['gene'] = self.df.index
        # Add prior values to the dataframe
        self.df['mL_mean'] = self.mL
        self.df['mL_stdev'] = self.mL_std
        self.df['sL_mean'] = self.sL
        self.df['sL_stdev'] = self.sL_std
        self.df['tI_mean'] = self.tI
        self.df['tI_stdev'] = self.tI_std
        if self.ET_sense is True:
            self.df['mT_mean'] = self.mT
            self.df['mT_stdev'] = self.mT_std
            self.df['sT_mean'] = self.sT
            self.df['sT_stdev'] = self.sT_std
            self.df['w_E_mean'] = self.w_E
            self.df['w_E_stdev'] = self.w_E_std
            self.df['w_T_mean'] = self.w_T
            self.df['w_T_stdev'] = self.w_T_std
        self.df['w_LI_mean'] = self.w_LI
        self.df['w_LI_stdev'] = self.w_LI_std
        self.df['w_B_mean'] = self.w_B
        self.df['w_B_stdev'] = self.w_B_std
        if self.antisense is True:
            self.df['mL_a_mean'] = self.mL_a
            self.df['mL_a_stdev'] = self.mL_a_std
            self.df['sL_a_mean'] = self.sL_a
            self.df['sL_a_stdev'] = self.sL_a_std
            self.df['tI_a_mean'] = self.tI_a
            self.df['tI_a_stdev'] = self.tI_a_std
            self.df['w_aLI_mean'] = self.w_aLI
            self.df['w_aLI_stdev'] = self.w_aLI_std
            self.df['w_aB_mean'] = self.w_aB
            self.df['w_aB_stdev'] = self.w_aB_std
            if self.ET_antisense:
                self.df['mT_a_mean'] = self.mT_a
                self.df['mT_a_stdev'] = self.mT_a_std
                self.df['sT_a_mean'] = self.sT_a
                self.df['sT_a_stdev'] = self.sT_a_std
                self.df['w_aE_mean'] = self.w_aE
                self.df['w_aE_stdev'] = self.w_aE_std
                self.df['w_aT_mean'] = self.w_aT
                self.df['w_aT_stdev'] = self.w_aT_std
        # get the percentiles if they exist
        if self.percentiles:
            percentile_dict = self.percentile_extract()
            # save each percentile as per_pos and per_neg
            for perc, perc_list in percentile_dict.items():
                self.df[perc] = perc_list
        # Add the + & - strand coverage, and elbow range
        # initiate lists
        pos_cov_list = []
        neg_cov_list = []
        elbo_lrange_list = []
        elbo_urange_list = []
        fit_time_list = []
        
        # iterate through log items to get values
        for gene in self.df['gene']:
            if len(self.log[gene]) > 1:
                pos_cov_list = pos_cov_list + [self.log[gene]['strand_cov'][0]]
                neg_cov_list = neg_cov_list + [self.log[gene]['strand_cov'][1]]
                elbo_lrange_list = elbo_lrange_list + [self.log[gene]['elbo_range'][0]]
                elbo_urange_list = elbo_urange_list + [self.log[gene]['elbo_range'][1]]
                fit_time_list = fit_time_list + [self.log[gene]['fit_time_min']]
            else:
                pos_cov_list = pos_cov_list + ["NA"]
                neg_cov_list = neg_cov_list + ["NA"]
                elbo_lrange_list = elbo_lrange_list + ["NA"]
                elbo_urange_list = elbo_urange_list + ["NA"]
                fit_time_list = fit_time_list + ["NA"]
        # actually add to dataframe
        self.df['pos_cov'] = pos_cov_list
        self.df['neg_cov'] = neg_cov_list
        self.df['elbo_lrange'] = elbo_lrange_list
        self.df['elbo_urange'] = elbo_urange_list
        self.df['fit_time_min'] = fit_time_list                       
        # Sort the dataframe based on geneid
        self.df.sort_index(inplace=True)
        return self.df
 

class FitParse_old:
    '''
    ORIGINAL: FitParse loads the results of a LIET fitting run from the standard output 
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
        self.mT, self.mT_std = self.param_extract('mT', stdev=True)
        self.sT, self.sT_std = self.param_extract('sT', stdev=True)
        self.w, self.w_std = self.param_extract('w', stdev=True)
        self.mL_a, self.mL_a_std = self.param_extract('mL_a', stdev=True)
        self.sL_a, self.sL_a_std = self.param_extract('sL_a', stdev=True)
        self.tI_a, self.tI_a_std = self.param_extract('tI_a', stdev=True)
        self.w_a, self.w_a_std = self.param_extract('w_a', stdev=True)

        # Parse strand coverage and min/max elbo values from log file
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
                        field, value = line.strip().split(':')
                        if field == 'fit_range':
                            value = value.strip('()').split(',')
                            value = tuple(map(int, value))
                        elif field == 'strand_cov':
                            value = value.strip('()').split(',')
                            value = tuple(map(int, value))
                        elif field == 'elbo_range':
                            value = value.strip('()').split(',')
                            value = tuple(map(float, value))
                        elif field == 'fit_time_min':
                            value = float(value)
                        else:
                            continue
                        self.log[gene_id].update({field: value})
            
            self.cov_pos = []
            self.cov_neg = []

            for g in self.genes:
                self.cov_pos.append(self.log[g]['strand_cov'][0])
                self.cov_neg.append(abs(self.log[g]['strand_cov'][1]))


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


            

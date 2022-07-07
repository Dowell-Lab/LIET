import sys
from collections import OrderedDict

class FitParse:
    '''
    FitParse loads the results of a LIET fitting run from the standard output 
    file. Results are organzed into several dictionaries and lists for easier 
    parsing. Additionally, the class contains methods for parsing the main 
    results dictionary <fits>.
    '''

    def __init__(self, res_file):

        self.definitions = OrderedDict({
            "mL": "Sense strand loading position (mu)",
            "sL": "Sense strand loading stdev (sigma)",
            "tI": "Sense strand initiation length (tau)",
            "mT": "Sense strand termination position (mu)",
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
                
                # Check line has a fit result
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

    
    def mL(self, stdev=False):
        '''
        Extract mL values (ordered based on genes list) from fits dictionary 
        and output them to a list.
        '''
        ml_vals = []
        if stdev:
            ml_stdev = []

        for g in self.genes:
            ml_vals.append(self.fits[g]['mL'][0])
            if stdev:
                ml_stdev.append(self.fits[g]['mL'][1])

        if stdev:
            return ml_vals, ml_stdev
        else:
            return ml_vals
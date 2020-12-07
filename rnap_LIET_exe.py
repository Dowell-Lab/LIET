import numpy as np

from rnap_lib_LIET_class import LIET
import rnap_lib_data_proc as dp
import rnap_lib_data_sim as ds

## PARSE INPUT FILES ##########################################################

# Load config (contents: FILES, MODEL, PRIORS, DATA_PROC, FITTING, RESULTS)
config_file = "C:\\Users\\Jacob\\Dropbox\\0DOWELL\\rnap_model\\LIET_test\\LIET_test_config.txt"
config = dp.config_loader(config_file)

# Parse annotation file
annot_file = config['FILES']['ANNOTATION']
annot_dict = dp.annot_loader(annot_file)

# Open results and log files

# Open bedgraph file
bg_file = config['FILES']['BEDGRAPH']
with open(bg_file, 'r') as bg:

    ## LOOP OVER ANNOTATIONS ##################################################
    for chromosome, annotations in annot_dict.items():

        for region, gene_id in annotations.items():
            
            start = region[0]
            stop = region[1]
            strand = region[2]

            # Determine pad
            pad = config['DATA_PROC']['PAD']
            if isinstance(pad, float):
                pad = round((stop - start) * pad)

            # Extract reads from bedgraph file for region
            begin = start - pad
            end = stop + pad
            current_line, preads, nreads = dp.bgreads(bg, begin, end)

            # Convert reads to list
            pread_list = []
            nread_list = []
            for position, count in preads.items():
                pread_list.extend([position]*count)
            for position, count in nreads.items():
                nread_list.extend([position]*count)

            # Generate model object
            liet = LIET()

            # Load annotation into LIET class object
            annot = {
                'gene_id': gene_id, 
                'chrom': chromosome, 
                'start': start, 
                'stop': stop, 
                'strand': strand
            }
            liet.load_annotation(**annot)

            # Load read data into LIET class object
            coordinates = list(range(begin, end))
            data = {
                'positions': coordinates, 
                'pos_reads': pread_list, 
                'neg_reads': nread_list, 
                'pad': pad, 
                'shift': config['DATA_PROC']['RANGE_SHIFT']
            }
            liet.load_seq_data(**data)

            # Load priors
            if config['DATA_PROC']['RANGE_SHIFT']:
                tss = 0
                tts = abs(stop-start)
            else:
                tss = start
                tts = stop
            priors = dp.prior_config(config['PRIORS'], tss, tts)

            liet.set_priors(**priors)

            # Build model
            liet.build_model(
                antisense=config['MODEL']['ANTISENSE'],
                background=config['MODEL']['BACKGROUND']
            )

            # Fit
            
# Summarize posteriors
# Evaluate "best fit" values
# Evaluate whether or not to refit

# Refit OR record results to results file


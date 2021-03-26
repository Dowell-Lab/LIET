import sys
import numpy as np

from rnap_lib_LIET_class import LIET
import rnap_lib_data_proc as dp
import rnap_lib_data_sim as ds
import rnap_lib_fitting_results as fr

## PARSE INPUT FILES ==========================================================

# Load config (contents: FILES, MODEL, PRIORS, DATA_PROC, FITTING, RESULTS)
config_file = "C:\\Users\\Jacob\\Dropbox\\0DOWELL\\rnap_model\\LIET\\test_config.txt"
config = dp.config_loader(config_file)

# Parse annotation file
annot_file = config['FILES']['ANNOTATION']
annot_dict = dp.annot_loader(annot_file)

for k, v in annot_dict.items():
    print(k, '\t', v)
    
# Open results and log files and initialize them
res_filename = config['FILES']['RESULTS']
res_file = open(res_filename, 'w')
fr.res_file_init(res_file, config_file) 

log_filename = f"{res_filename}.log"                                     # Need to finalize this function
log_file = open(log_filename, 'w')
fr.log_file_init(log_file)

# Open bedgraph file
bg_file = config['FILES']['BEDGRAPH']

with open(bg_file, 'r') as bg:
    current_bgl = ['chr1', 0, 0, 0]    # Initialize current line
    ## MAIN LOOP OVER ANNOTATIONS =========================================
    for chromosome, annotations in annot_dict.items():
        print(chromosome)
        for region, gene_id in annotations.items():
            
            print(f'Running: {gene_id}')
            start, stop, strand = region
            
            # Compute padded range
            pad = config['DATA_PROC']['PAD']
            pad_args = [start, stop, strand, pad]
            adj_start, adj_stop = dp.pad_calc(*pad_args)
            # Extract reads
            bg_args = [bg, current_bgl, chromosome, adj_start, adj_stop]
            print(f"BG_ARGS: {bg_args[1:]}")
            current_bgl, preads, nreads = dp.bgreads(*bg_args)
            # Convert read dict to list
            preads_list = np.array(dp.reads_d2l(preads))
            nreads_list = np.array(dp.reads_d2l(nreads))

            print(f"Range: {adj_start}, {adj_stop}, {strand}")

            try:
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
                print(annot)
            except:
                print("Error load annotation")
            try:
                # Load read data into LIET class object
                coordinates = np.array(range(adj_start, adj_stop))
                data = {
                    'positions': coordinates, 
                    'pos_reads': preads_list, 
                    'neg_reads': nreads_list, 
                    'pad': pad, 
                    'shift': config['DATA_PROC']['RANGE_SHIFT']
                }
                liet.load_seq_data(**data)
            except:
                print("Error loading data")
                print("PAD: ", pad)
                print(f"READS: {preads_list[0:10]}, {nreads_list[0:10]}")
            try:
                # Load priors
                if config['DATA_PROC']['RANGE_SHIFT']:
                    tss = 0
                    tcs = abs(stop-start)
                else:
                    tss = start
                    tcs = stop
                priors = dp.prior_config(config['PRIORS'], tss, tcs)

                liet.set_priors(**priors)
            except:
                print("Error setting priors")
            try:
                # Build model
                liet.build_model(
                    antisense=config['MODEL']['ANTISENSE'],
                    background=config['MODEL']['BACKGROUND']
                )
                print('Running fit...')
                # Fit
                fit = fr.vi_fit(
                    liet.model,
                    method=config['FIT']['METHOD'],
                    optimizer=config['FIT']['OPTIMIZER'],
                    learning_rate=config['FIT']['LEARNING_RATE'],
                    start=None,                                                 # NEEDS IMPLEMENTATION
                    iterations=config['FIT']['ITERATIONS'],
                    tolerance=config['FIT']['TOLERANCE'],                       # NEED TO FIX IMPLEMENTATION
                    param_tracker=False,                                        # NEEDS IMPLEMENTATION
                )

            ## Currently omitting these steps =============================
            # Evaluate "best fit" values
            # Evaluate whether or not to refit 
            # Run refit if convergence criteria not met
            ## ============================================================
                print("Summarizing post...")
                # Summarize posteriors
                post_stats = fr.posterior_stats(
                    fit['approx'],
                    N=config['RESULTS']['SAMPLES'],
                    calc_mean=config['RESULTS']['MEAN'],
                    calc_mode=config['RESULTS']['MODE'],
                    calc_median=config['RESULTS']['MEDIAN'],
                    calc_stdev=config['RESULTS']['STDEV'],
                    calc_skew=config['RESULTS']['SKEW']
                )

                # Record results of fitting
                res = fr.results_format(gene_id, post_stats, stat='mean')
                res_file.write(res)

                fr.log_write(log_file, liet, fit)
            except:
                res_file.write(f"{gene_id}\t{start}\t{stop}\t{strand}\tERROR\n")
#except:
#    res_file.write("RUN ERROR")
#    res_file.close()
#    sys.exit(1)
    
res_file.close()
log_file.close()
sys.exit(0)
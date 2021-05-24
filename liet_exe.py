import sys
import time
import argparse
import numpy as np

from rnap_lib_LIET_class import LIET
import rnap_lib_data_proc as dp
import rnap_lib_data_sim as ds
import rnap_lib_fitting_results as fr
import rnap_lib_plotting as pl

## COMMAND LINE INFO ==========================================================
description_text = ("LIET model executable. Required input: config file.")

config_text = ("Path to config file.")

parser = argparse.ArgumentParser(description = description_text)

## PARSE ARGS AND OPEN CONFIG =================================================
parser.add_argument('-c', '--config', type = str, help = config_text)
args = parser.parse_args()
config_file = args.config

## PROCESS INPUTS =============================================================

# Load config (contents: FILES, MODEL, PRIORS, DATA_PROC, FITTING, RESULTS)
#config_file = "C:\\Users\\Jacob\\Dropbox\\0DOWELL\\rnap_model\\LIET\\test_config.txt"
config = dp.config_loader(config_file)

# Parse annotation file
annot_file = config['FILES']['ANNOTATION']
annot_dict, pad_dict = dp.annot_loader(annot_file)

#for k, v in annot_dict.items():
#    print(k, '\t', v)
    
# Open results and log files and initialize them
res_filename = config['FILES']['RESULTS']
res_file = open(res_filename, 'w')
fr.res_file_init(res_file, config_file) 

log_filename = f"{res_filename}.log"                                     # Need to finalize this function
log_file = open(log_filename, 'w')
fr.log_file_init(log_file, config_file)

# Open bedgraph files
bgp_file = config['FILES']['BEDGRAPH_POS']
bgn_file = config['FILES']['BEDGRAPH_NEG']

with open(bgp_file, 'r') as bgp, open(bgn_file, 'r') as bgn:
    current_bgpl = ['chr1', 0, 0, 0]    # Initialize current line
    current_bgnl = ['chr1', 0, 0, 0]    # Initialize current line
    ## MAIN LOOP OVER ANNOTATIONS =========================================
    for chromosome, annotations in annot_dict.items():

        for region, gene_id in annotations.items():                     # I SHOULD CHECK THIS IS SORTED
            
            start_time = time.time()
            print(f'\nRunning: {gene_id}')
            start, stop, strand = region
            
            # Compute padded range
#            pad = config['DATA_PROC']['PAD']
            pad = pad_dict[gene_id]
            pad_args = [start, stop, strand, pad]
            adj_start, adj_stop = dp.pad_calc(*pad_args)

            # Extract reads
            bgp_args = [bgp, current_bgpl, chromosome, adj_start, adj_stop]
            current_bgpl, preads = dp.bgreads(*bgp_args)
            
            bgn_args = [bgn, current_bgnl, chromosome, adj_start, adj_stop]
            current_bgnl, nreads = dp.bgreads(*bgn_args)

            # Convert read dict to list
            preads_list = np.array(dp.reads_d2l(preads))
            nreads_list = np.array(dp.reads_d2l(nreads))

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
            except:
                print("Error load annotation")
                continue

            try:
                # Load read data into LIET class object
                coordinates = np.array(range(adj_start, adj_stop))
                data = {
                    'coord': coordinates, 
                    'pos_reads': preads_list, 
                    'neg_reads': nreads_list, 
                    'pad': pad, 
                    'shift': config['DATA_PROC']['RANGE_SHIFT']
                }
                liet.load_seq_data(**data)
            except:
                print("Error loading data")
                continue

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
                continue

            try:
                # Build model
                liet.build_model(
                    antisense=config['MODEL']['ANTISENSE'],
                    background=config['MODEL']['BACKGROUND']
                )
#                print('Running fit...')
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
#                print("Summarizing post...")
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

#                print(f"post stats: {post_stats}")
#                print(f"annot: {liet.data['annot']}")

                # Record results of fitting
                res = fr.results_format(
                    liet.data['annot'], 
                    post_stats, 
                    stat='mean'
                )
                res_file.write(res)

                # Log meta info for fit
                fr.log_write(log_file, liet, fit)

            except:
                res_file.write(f"{chromosome}\t{start}\t{stop}\t{strand}\t{gene_id}\tERROR\n")
                continue

            # Log fitting time
            end_time = time.time()
            fit_time = np.around((end_time - start_time) / 60, 2)
            log_file.write(f"fit_time_min:{fit_time}\n")

            # Plot fit result
            try:
                liet.results = post_stats
#                if liet.data['shift'] == True:
#                    coord = liet.data['coord'] - liet.data['annot']['start']
#                else:
#                    coord = liet.data['coord']
#                cmin = coord.min()
#                cmax = coord.max()
#                bins = np.linspace(cmin, cmax, 2000)

#                print(f"MIN/MAX: {cmin}/{cmax}")

                lplot = pl.LIET_plot(
                    liet, 
#                    bins = 'auto',
                    data=True,
                    antisense=True,
                    sense=True,
                    save=f"liet_plot_{gene_id}.pdf"
                )

            except:
                print(f"Can't plot fit result for {gene_id}")
            
#except:
#    res_file.write("RUN ERROR")
#    res_file.close()
#    sys.exit(1)

res_file.close()
log_file.close()
sys.exit(0)

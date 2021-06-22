import sys
import time
import argparse
import numpy as np
import multiprocessing as mp

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

# Open bedgraph files and load reads
bgp_file = config['FILES']['BEDGRAPH_POS']
bgn_file = config['FILES']['BEDGRAPH_NEG']
reads_dict = dp.bedgraph_loader(bgp_file, bgn_file, annot_dict, pad_dict)

print(f"RD: {reads_dict.keys()}")
# Combine annot_dict and reads_dict for parallelization input
mpargs = {
    (ch, a, gid) : reads_dict[gid] 
    for ch, annots in annot_dict.items() 
    for a, gid in annots.items()
}

# Parallelizable function that performs all the fitting
def fit_routine(fit_instance, config, pad_dict):
    
    start_time = time.time()

    annot, reads = fit_instance

    chrom = annot[0]
    start, stop, strand = annot[1]
    gene_id = annot[2]

    pad = pad_dict[gene_id]
    pad_args = [start, stop, strand, pad]
    adj_start, adj_stop = dp.pad_calc(*pad_args)

#    return_dict = {'res': f"{gene_id}\tERROR", 'log': f">{gene_id}:ERROR"}
    return_dict = {'res': f"{gene_id}\n", 'log': (f">{gene_id}\n", )}

    # Convert read dict to list
    preads_list = np.array(dp.reads_d2l(reads[0]))
    nreads_list = np.array(dp.reads_d2l(reads[1]))

    print(f"RUNNING: {gene_id}")

    try:
        # Generate model object
        liet = LIET()

        # Load annotation into LIET class object
        annot_dict = {
            'gene_id': gene_id, 
            'chrom': chrom, 
            'start': start, 
            'stop': stop, 
            'strand': strand
        }
        liet.load_annotation(**annot_dict)
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: annot load error\n"
        return {annot: return_dict}

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
        return_dict['res'] = f"{annot_dict['gene_id']}: seq data load error\n"
        return {annot: return_dict}

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
        return_dict['res'] = f"{annot_dict['gene_id']}: prior set error\n"
        return {annot: return_dict}

    try:
        # Build model
        liet.build_model(
            antisense=config['MODEL']['ANTISENSE'],
            background=config['MODEL']['BACKGROUND']
        )
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: model build error\n"
        return {annot: return_dict}

    try:
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
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: fitting error\n"
        return {annot: return_dict}

    ## Currently omitting these steps =============================
    # Evaluate "best fit" values
    # Evaluate whether or not to refit 
    # Run refit if convergence criteria not met
    ## ============================================================
    #                print("Summarizing post...")
        # Summarize posteriors
    try:
        post_stats = fr.posterior_stats(
            fit['approx'],
            N=config['RESULTS']['SAMPLES'],
            calc_mean=config['RESULTS']['MEAN'],
            calc_mode=config['RESULTS']['MODE'],
            calc_median=config['RESULTS']['MEDIAN'],
            calc_stdev=config['RESULTS']['STDEV'],
            calc_skew=config['RESULTS']['SKEW']
        )
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: post stat error\n"
        return {annot: return_dict}

    try:
        # Record results of fitting
        res_string = fr.results_format(
            liet.data['annot'], 
            post_stats, 
            stat='mean'
        )
        return_dict['res'] = res_string
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: res str error\n"
        return {annot: return_dict}

    try:
        # Log meta info for fit
        log_strings = fr.log_format(liet, fit)

        end_time = time.time()
        fit_time = np.around((end_time - start_time) / 60, 2)
        time_string = f"fit_time_min:{fit_time}\n"
        
        log_strings = (*log_strings, time_string)

        return_dict['log'] = log_strings
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: log str error\n"
        return {annot: return_dict}

    # Plot fit result
    try:
        liet.results = post_stats
        lplot = pl.LIET_plot(
            liet, 
            data=True,
            antisense=True,
            sense=True,
            save=f"liet_plot_{gene_id}.pdf"
        )
        lplot.close()
    except:
        print(f"Can't plot fit result for {gene_id}")

    return {annot: return_dict}

# Run fitting in parallel
#pool = mp.Pool(mp.cpu_count())
print(f"CPU: {mp.cpu_count()}")
#res = pool.starmap(fit_routine, [(i, config, pad_dict) for i in mpargs.items()])
#print(f"LENGTH res: {len(res)}")

with mp.Pool(processes=mp.cpu_count()) as pool:
    res = list(pool.apply(fit_routine, args=[i, config, pad_dict]) for i in mpargs.items())
#res = res.get()
#pool.close()

print("Fitting complete...")

# Convert `res` tuple into a dictionary
res_dict = {}
for i in res:
#    print(f"ref: {i}")
    res_dict.update(i)

# Open results and log files and initialize them
res_filename = config['FILES']['RESULTS']
res_file = open(res_filename, 'w')
fr.res_file_init(res_file, config_file)
print(f"res file: {res_filename}")

log_filename = f"{res_filename}.log"                                     # Need to finalize this function
log_file = open(log_filename, 'w')
fr.log_file_init(log_file, config_file)
print(f"log file: {log_filename}")

# Record results and log information
for annot, fitres in res_dict.items():
    res_str = fitres['res']
    print(f"RES LINE: {res_str}\n")
    log_str = fitres['log']
    print(f"LOG LINE: {res_str}\n")
    try:
        res_file.write(res_str)
    except:
        print(f"Can't write res: {annot}")
    try:
        for line in log_str:
            log_file.write(line)
    except:
        print(f"Can't write log: {annot}")

print("ALL DONE")
res_file.close()
log_file.close()

sys.exit(0)
import sys
import os
import time
import argparse
import traceback
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from rnap_lib_LIET_class import LIET
import rnap_lib_data_proc as dp
import rnap_lib_data_sim as ds
import rnap_lib_fitting_results as fr
import rnap_lib_plotting as pl


# Input processing function (unlike fit_routine() this is not paralellized)
def input_processing(config_file):
    '''
    Using the config file, this function generates the reads and annotations 
    dictionaries over which the fit routine is parallelized. The reads/annots
    outputs are filtered for coverage.

    Parameters
    ----------
    config_file : string
        Full path to LIET config file.

    Returns
    -------
    config : dict
        Contains all the input data from the config file.

    pad_dict : dict
        Contains 5' and 3' end padding lengths for each input gene. These are
        not the default values. Those are contained in the config file.
        Format: {'gene ID': (pad5, pad3), ...}

    reads_dict : dict
        Dictionary containing both the positive and negative strand read 
        counts for each gene in the annotations. Read objects are Counter-like 
        dictionaries.
        Format: {'gene_ID': (preads, nreads), ...}

    annot_dict : dict
        Contains the chrom/start/stop/strand for each input gene. Sorted by 
        chromosome and start/stop values.
        Format: {'chrom': {(start, stop, strand): gene_id, ...}, ...}
    
    filter_log : dict
        Dictionary containing the coverage values for those genes which have 
        been filtered out due to coverage filtering.
        Format: {"gene_ID": "<log string with sense/antisense counts>", ...}
    '''
    # Load config (contents: FILES, MODEL, PRIORS, DATA_PROC, FITTING, RESULTS)
    config = dp.config_loader(config_file)

    # Parse annotation file
    annot_file = config['FILES']['ANNOTATION']
    annot_dict = dp.annot_BED6_loader(annot_file)

    # Extract all gene ID's from annotation dictionary
    gene_id_list = [gid for ch, rois in annot_dict.items() 
                    for gid in rois.values() ]

    # Compute padding dictionary from file and default pads
    default_pads = config['DATA_PROC']['PAD']  # (5'pad, 3'pad)
    pad_file = config['FILES']['PAD_FILE']
    pad_dict = dp.pad_dict_generator(gene_id_list, default_pads, pad_file)

    # Open bedgraph files and load reads
    bgp_file = config['FILES']['BEDGRAPH_POS']
    bgn_file = config['FILES']['BEDGRAPH_NEG']
    reads_dict = dp.bedgraph_loader(bgp_file, bgn_file, annot_dict, pad_dict)

    # Filter out genes with zero/low coverage
    if config['DATA_PROC']['COV_THRESHOLDS']:
        cov_thresh = config['DATA_PROC']['COV_THRESHOLDS']
    else:
        cov_thresh = (0, 0)
    filtered = dp.cov_filter(reads_dict, annot_dict, thresholds=cov_thresh)
    reads_dict, annot_dict, filter_log = filtered

    return config, pad_dict, reads_dict, annot_dict, filter_log


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

    return_dict = {
        'res': f"{gene_id}\n", 
        'log': (f">{gene_id}\n", ), 
        'err': [f">{gene_id}\n"]
    }

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
        return_dict['err'].append(f"{traceback.format_exc()}\n")
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
        return_dict['res'] = f"{annot_dict['gene_id']}: seq data error\n"
        return_dict['err'].append(f"{traceback.format_exc()}\n")
        return {annot: return_dict}

    try:
        # Load priors
        if config['DATA_PROC']['RANGE_SHIFT']:
            tss = 0
            tcs = abs(stop-start)
        else:
            tss = start
            tcs = stop
        # NOTE: prior_config will acount for gene length in mT offset for 
        # RANGE_SHIFT == True.
        priors = dp.prior_config(config['PRIORS'], tss, tcs)
        liet.set_priors(**priors)
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: prior set error\n"
        return_dict['err'].append(f"{traceback.format_exc()}\n")
        return {annot: return_dict}

    try:
        # Build model
        liet.build_model(
            antisense=config['MODEL']['ANTISENSE'],
            background=config['MODEL']['BACKGROUND']
        )
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: model error\n"
        return_dict['err'].append(f"{traceback.format_exc()}\n")
        return {annot: return_dict}

    try:
        # Fit
        fit = fr.vi_fit(
            liet.model,
            method=config['FIT']['METHOD'],
            optimizer=config['FIT']['OPTIMIZER'],
            learning_rate=config['FIT']['LEARNING_RATE'],
            start=None,
            iterations=config['FIT']['ITERATIONS'],
            tolerance=config['FIT']['TOLERANCE'],
            param_tracker=False,
        )
    except:
        return_dict['res'] = f"{annot_dict['gene_id']}: fitting error\n"
        return_dict['err'].append(f"{traceback.format_exc()}\n")
        return {annot: return_dict}

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
        return_dict['err'].append(f"{traceback.format_exc()}\n")
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
        return_dict['err'].append(f"{traceback.format_exc()}\n")
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
        return_dict['err'].append(f"{traceback.format_exc()}\n")
        return {annot: return_dict}

    # Plot fit result
    if config['RESULTS']['PDF']:
        try:
            # Add posterior stats to liet object before plotting.
            liet.results = post_stats
            lplot = pl.LIET_plot(
                liet, 
                data=True,
                antisense=True,
                sense=True,
                save=config['RESULTS']['PDF']
                #save=f"liet_plot_{gene_id}.pdf"
            )
            plt.close(lplot)
        except:
            print(f"Can't plot fit result for {gene_id}")
            return_dict['err'].append(f"{traceback.format_exc()}\n")
    else:
        pass
    
    return {annot: return_dict}


### MAIN FUNCTION #############################################################
def main():

    ## SLURM INFO =============================================================
    env = os.environ
    cpu_num = int(os.environ['SLURM_CPUS_ON_NODE'])

    ## COMMAND LINE INFO ======================================================
    description_text = ("LIET model executable. Required input: config file.")

    config_text = ("Path to config file.")

    parser = argparse.ArgumentParser(description = description_text)

    ## PARSE ARGS AND OPEN CONFIG =============================================
    parser.add_argument('-c', '--config', type = str, help = config_text)
    args = parser.parse_args()
    config_file = args.config

    ## PROCESS INPUTS =========================================================

    inputs = input_processing(config_file)
    config, pad_dict, reads_dict, annot_dict, filter_log = inputs
    #print(f"RD: {list(reads_dict.keys())}")
    
    # Combine annot_dict and reads_dict for parallelization input
    mpargs = {
        (ch, a, gid) : reads_dict[gid] 
        for ch, annots in annot_dict.items() 
        for a, gid in annots.items()
    }

    # RUN PARALLEL FITTING ====================================================
    print(f"CPU: {cpu_num}")

    pool = mp.Pool(cpu_num)
    res = pool.starmap(
        fit_routine, 
        [(i, config, pad_dict) for i in mpargs.items()]
    )
    pool.close()

    print("Fitting complete")

    # Convert 'res' tuple into a dictionary
    res_dict = {}
    for i in res:
    #    print(f"ref: {i}")
        res_dict.update(i)

    # Open err function
    res_filename = config['FILES']['RESULTS']
    err_filename = res_filename + ".err"
    err_file = open(err_filename, 'w')

    # Open results and log files and initialize them
    res_filename = config['FILES']['RESULTS']
    res_file = open(res_filename, 'w')
    fr.res_file_init(res_file, config_file)
    print(f"res file: {res_filename}")

    log_filename = f"{res_filename}.log"
    log_file = open(log_filename, 'w')
    fr.log_file_init(log_file, config_file)
    print(f"log file: {log_filename}")

    # Record results and log information
    # annot format: (chrom, (start, stop, strand), gene_id)
    for annot, fitres in res_dict.items():
        res_str = fitres['res']
        print(f"RES LINE: {res_str}\n")
        log_str = fitres['log']
        print(f"LOG LINE: {log_str}\n")
        err_str = fitres['err']

        try:
            res_file.write(res_str)
        except:
            print(f"Can't write res: {annot}")
        try:
            for line in log_str:
                log_file.write(line)
        except:
            print(f"Can't write log: {annot}")
        try:
            # Record fitting error
            for line in err_str:
                err_file.write(line)
            # Record coverage filter error
            if annot[2] in filter_log.keys():
                err_file.write(filter_log[annot[2]])
        except:
            print(f"Can't write err: {annot}")

    res_file.close()
    log_file.close()
    err_file.close()

    sys.exit(0)


if __name__ == '__main__':
    main()
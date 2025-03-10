import numpy as np
import scipy as sp
import pymc as pm
import time
from collections import OrderedDict

import rnap_lib_data_proc as dp
from rnap_lib_data_sim import invert, gene_model
from liet_res_class import FitParse


    #VI options: ADVI, NFVI, FullRankADVI, NFVI, SVGD, KLqp

    # optimizer options:     
    #"sgd",
    #"momentum",
    #"nesterov_momentum",
    #"adagrad",
    #"adagrad_window",
    #"rmsprop",
    #"adadelta",
    #"adam",
    #"adamax",
    #"norm_constraint",
    #"total_norm_constraint",

def vi_fit(
    model,
    method='advi',
    optimizer='adamax',
    learning_rate=None,
    start=None,
    iterations=50000,
    tolerance=None,
    param_tracker=False,
    prog_bar=False
):
    '''
    Function that performs fitting using the VI framework. Returns inference 
    object and approximation, as well as tracker object, if specified.

    Parameters
    ----------
    model : pymc3 model object
        ...

    method : str
        ...

    optimizer : str
        ...

    learning_rate : float
        ...
    
    start : dict
        ...

    iterations : int
        ...

    tolerance : float
        ...

    param_tracker : bool
        ...


    Returns
    -------
    fit : dict
        ...
        
    '''
    fit = {
        'vi': None,
        'approx': None,
        'tracker': None
    }

    # Optimizer selection
    optimizer = optimizer.lower()
    if optimizer == 'adadelta':
        opt = pm.adadelta()
    elif optimizer == 'adagrad':
        opt = pm.adagrad()
    elif optimizer == 'adagrad_window':
        opt = pm.adagrad_window()
    elif optimizer == 'adam':
        opt = pm.adam()
    elif optimizer == 'adamax':
        opt = pm.adamax()
    elif optimizer == 'momentum':
        opt = pm.momentum()
    elif optimizer == 'rmsprop':
        opt = pm.rmsprop()
    elif optimizer == None:
        opt = None
    else:
        raise ValueError(f'Optimizer {optimizer} not an available option.')

    # Update learning rate
    if learning_rate:
        opt(learning_rate = learning_rate)

    # Method selection and VI initialization
    method = method.lower()
    if method == 'advi':
        vi = pm.ADVI(model=model, start=start)
    elif method == 'nfvi':
        vi = pm.NFVI(model=model, start=start)
    elif method == 'fullrankadvi':
        vi = pm.FullRankADVI(model=model, start=start)
    else:
        raise ValueError(f'Method {method} not an available option.')
    
    # Define fit dictionary
    fit_dict = {'obj_optimizer': opt}
    callbacks = []

    # Set convergence tolerance
    if tolerance:
        check_conv = pm.callbacks.CheckParametersConvergence(
            every=100,
            diff='absolute',
            tolerance=tolerance
        )
        callbacks.append(check_conv)

    # Include parameter tracking
    if param_tracker and method in ['advi', 'fullrankadvi']:
        tracker = pm.callbacks.Tracker(
            mean=vi.approx.mean.eval,
            std=vi.approx.std.eval
        )
        callbacks.append(tracker)
    else:
        tracker = None
    
    # Add callbacks to fitting dictionary
    if callbacks:
        fit_dict['callbacks'] = callbacks

    # Run fit
    approx = vi.fit(iterations, progressbar=prog_bar, **fit_dict)

    # Add inference, approx, and tracker objects to output dict
    fit['vi'] = vi
    fit['approx'] = approx
    if tracker:
        fit['tracker'] = tracker

    return fit



#==============================================================================

def posterior_stats(
    post_approx,
    N=10000,
    calc_mean=True,
    calc_mode=False,
    calc_median=False,
    calc_stdev=True,
    calc_skew=False
):
    '''
    Compute specified stats for the posterior samples.

    Parameters
    ----------
    post_approx : pymc3 variational approximation object
        The object that results from running the '.fit()' method on the pymc3 
        variational object (produced by vi_fit() function). Must possess a 
        '.sample()' method.

    N : int
        Number of samples to generate from the posterior approximation.

    calc_mean : bool
        Indicate whether or not to return the mean of the posteriors.
    
    calc_mode : bool
        Indicate whether or not to return the mode of the posteriors.

    calc_median : bool
        Indicate whether or not to return the median of the posteriors.

    calc_stdev : bool
        Indicate whether or not to return the standard deviation of the 
        posteriors.
    
    calc_skew : bool
        Indicate whether or not to return the skew of the posteriors.


    Returns
    -------
        post_stats : dict
            Dictionary containing the specified stats for each of the params.
    '''
    # Sample approximation (convert arviz InferenceData obj to dict)
    posterior_samples = post_approx.sample(N).to_dict()
    posterior_samples = posterior_samples['posterior']

    # Filter out transformed variable names (those ending with '0')
    params = [e for e in posterior_samples.keys() if e[-1] != '0']

    # Mode computing function that uses guassian kde to smooth empirical dist
    def kde_mode(samples, tol=10):
        '''
        Computes the mode of the distribution after applying a guassian kde
        '''
        modes = np.array([])

        if np.size(np.shape(samples)) > 1:
            for col_samp in samples.T:

                smin = min(col_samp)
                smax = max(col_samp)
                N = int((smax - smin)/tol)
                
                samp_kde = sp.stats.gaussian_kde(col_samp)
                x = np.linspace(smin, smax, N)
                y = samp_kde.pdf(x)

                mode_val = max(zip(x,y), key = lambda x : x[1])[0]
                modes = np.append(modes, mode_val)
        
        else:
            smin = min(samples)
            smax = max(samples)
            N = int((smax - smin)/tol)
            
            samp_kde = sp.stats.gaussian_kde(samples)
            x = np.linspace(smin, smax, N)
            y = samp_kde.pdf(x)

            mode = max(zip(x,y), key = lambda x : x[1])[0]
            modes = np.append(modes, mode)

        return mode

# Compute the stats for parameters in params
    post_stats = {}
    for p in params:

        samps = posterior_samples[p][0,:]

        if calc_mean:
            mean_val = np.mean(samps, axis=0)
            
        else:
            mean_val = None

        if calc_median:
            median_val = np.median(samps, axis=0)
        else:
            median_val = None

        if calc_stdev:
            stdev_val = sp.stats.tstd(samps, axis=0)
        else:
            stdev_val = None

        if calc_mode:
            mode_val = kde_mode(samps)
        else:
            mode_val = None

        if calc_skew:
            skew_val = sp.stats.skew(samps, axis=0)
            skewtest_val = sp.stats.skewtest(samps, axis=0).pvalue
        else:
            skew_val = None
            skewtest_val = None

        post_stats[p] = {
            'mean': mean_val,
            'median': median_val,
            'stdev': stdev_val,
            'mode': mode_val,
            'skew': skew_val,
            'skewtest': skewtest_val
        }

    return post_stats



# RESULTS AND LOG FILE HANDLING ===============================================

def res_file_init(res_file, config_file_path):
    '''
    Parameters
    ----------
    res_file : python file object
        Results file that will contain the fit results.
    
    config_file_path : str
        Full path to the config file used for LIET run.

    Returns
    -------
    Null
    '''
    # Time stamp
    t=time.localtime()
    tformat = time.strftime("%H:%M:%S %d.%m.%Y", t)
    time_str = f"# {tformat}\n"
    res_file.write(time_str)

    # Config
    res_file.write(f"# CONFIG\t{config_file_path}\n")

    # Output format
    res_file.write("# Output format: param_name=value:stdev\n")

    res_file.write("#" + "="*79 + "\n")


def log_file_init(log_file, config_file_path):
    '''
    Parameters
    ----------
    log_file : python file object
        Log file associated with corresponding results file.
    
    config_file_path : str
        Full path to the config file used for LIET run.

    Returns
    -------
    Null
    '''
    # Time stamp
    t=time.localtime()
    tformat = time.strftime("%H:%M:%S %d.%m.%Y", t)
    time_str = f"# {tformat}\n"
    log_file.write(time_str)

    # Config
    log_file.write(f"# CONFIG\t{config_file_path}\n")

    log_file.write("#" + "="*79 + "\n")


def results_format(annot, post_stats, stat='mean', decimals=2):
    '''
    Parameters
    ----------
    NOTE: MUST UPDATE DOCSTRING
    annot : str
        Gene ID for which the fit results are being recorded

    post_stats : dict
        Dictionary returned by .posterior_stats() which contains the best fit 
        results for the parameters to be written to results file.

    stat : str
        Statistical value to include for each of the parameters. Must be one 
        of the following: 'mean', 'median', or 'mode'. Default: 'mean'

    Returns
    -------
    res : str
        Formatted string containing the gene ID and fit values for each 
        parameter.
    '''
    chrom = str(annot['chrom'])
    start = str(annot['start'])
    stop = str(annot['stop'])
    strand = str(annot['strand'])
    id = str(annot['gene_id'])
    fields = list([chrom, start, stop, strand, id])

    # Hard coding output order of parameters
    params = ['mL', 'sL', 'tI', 'mT', 'sT', 'w', 'mL_a', 'sL_a', 'tI_a', 'w_a']
    fit_res = []
    for p in params:

        # Extract param val depending on joint/indep priors for antisense
        if p in post_stats.keys():
            pvals = post_stats[p]
        elif p in ['mL_a', 'sL_a', 'tI_a']:
            p_alt = p.split('_')[0]
            pvals = post_stats[p_alt]
        else:
            raise ValueError(f'Parameter {p} should not be included in result')

        pval = np.around(pvals[stat], decimals=decimals)
        pstd = np.around(pvals['stdev'], decimals=decimals)

        # Enforce that weights sum to 1.00 (by adjusting the background wB)
        # Rounding sometimes results in sum being off by 1e-<decimals>
        if p == "w":
            wb_update = np.around(1.0 - sum(pval[0:3]), decimals=decimals)
            assert(abs(wb_update - pval[3]) <= 10**-decimals,
                "WARNING: Weight issue! wB rounding is not within tolerance."
            )
            pval[3] = wb_update

        pstring = f"{p}={pval}:{pstd}"
        fit_res.append(pstring)

    fit_res = ",".join(fit_res)
    fields.append(fit_res)

    res = "\t".join(fields) + "\n"
    return res


def log_write(log_file, liet, fit):
    '''
    Logs meta information about fit for each region.
    
    Parameters
    ----------
    log_file : python file object
        File to which logged info is written.

    liet : class
        LIET class object containing all the model info

    fit : dict
        Dictionary containing variation inference objects from pymc3

    Returns
    -------
    Null
    '''
    id = liet.data['annot']['gene_id']
    chrom = liet.data['annot']['chrom']
    start = liet.data['annot']['start']
    stop = liet.data['annot']['stop']
    strand = liet.data['annot']['strand']
    id_str = f">{id}:{chrom}:{start}:{stop}:{strand}\n"
    
    print(f"ID: {id_str}")

    rng = (min(liet.data['coord']), max(liet.data['coord']))
    rng_str =f"fit_range:{rng}\n"

    cov = (len(liet.data['pos_reads']), -1*len(liet.data['neg_reads']))
    cov_str = f"strand_cov:{cov}\n"

    elbo = (min(fit['vi'].hist), max(fit['vi'].hist))
    elbo_str = f"elbo_range:{elbo}\n"

    num_iter = len(fit['vi'].hist)
    iter_str = f"iterations:{num_iter}\n"

    # Write log strings
    log_file.write(id_str)
    log_file.write(rng_str)
    log_file.write(cov_str)
    log_file.write(elbo_str)
    log_file.write(iter_str)


def log_format(liet, fit):
    '''
    Logs meta information about fit for each region.
    
    Parameters
    ----------
    log_file : python file object
        File to which logged info is written.

    liet : class
        LIET class object containing all the model info

    fit : dict
        Dictionary containing variation inference objects from pymc3

    Returns
    -------
    Null
    '''
    id = liet.data['annot']['gene_id']
    chrom = liet.data['annot']['chrom']
    start = liet.data['annot']['start']
    stop = liet.data['annot']['stop']
    strand = liet.data['annot']['strand']
    id_str = f">{id}:{chrom}:{start}:{stop}:{strand}\n"
    
    print(f"ID: {id_str}")

    rng = (min(liet.data['coord']), max(liet.data['coord']))
    rng_str =f"fit_range:{rng}\n"

    cov = (len(liet.data['pos_reads']), -1*len(liet.data['neg_reads']))
    cov_str = f"strand_cov:{cov}\n"

    print(f"cov: {cov_str}")

    elbo = (min(fit['vi'].hist), max(fit['vi'].hist))
    elbo_str = f"elbo_range:{elbo}\n"

    print(f"elbo: {elbo_str}")

    num_iter = len(fit['vi'].hist)
    iter_str = f"iterations:{num_iter}\n"

    print(f"iters: {iter_str}")
    
    return id_str, rng_str, cov_str, elbo_str, iter_str


# POST FITTING RESULTS HANDLING ===============================================

def results_loader(gene_ids,
                   bedgraphs=None, 
                   config=None, 
                   result=None, 
                   log=None):
    '''
    This function uses much of the input data processing functionality to read 
    in LIET fitting results (from the .liet and .liet.log files) as well as 
    the read data, for the purposes of plotting the model fit. Uses the 
    following functions: config_loader(), bedgraph_loader(), FitParse, ...
    '''

    if config:
        # Parse config file
        config_parse = dp.config_loader(config)

        # Only need the input files from config
        bgp_file = config_parse['FILES']['BEDGRAPH_POS']
        bgn_file = config_parse['FILES']['BEDGRAPH_NEG']

    elif bedgraphs:
        assert isinstance(bedgraphs, (tuple, list)), "bedgraphs not a tuple"
        bgp_file, bgn_file = bedgraphs
    
    else:
        raise ValueError("You must specify either config or bedgraphs.")
    
    fit_parse = FitParse(result, log_file=log)

    # Determine chromosome string order
    chr_order = dp.chrom_order_reader(bgp_file, bgn_file)

    # Check all genes in gene_ids are contained in the fit result
    for gid in gene_ids:
        assert(gid in fit_parse.genes, f"Gene {gid} not in fit result.")

    # Build annotation dict for input into bedgraph_loader(). Does not assume 
    # FitParse.annotations is ordered by chrom or start position. Initialized 
    # from chr_order, checking against set of chromosomes in FitParse object.
    res_chr_set_tmp = set(np.array(list(fit_parse.annotations.values()))[:,0])
    annot_dict = OrderedDict([
        (chrom, {}) for chrom in chr_order.keys() 
        if chrom in res_chr_set_tmp
    ])
    for gid, annot in fit_parse.annotations.items():
        if gid in gene_ids:
            annot_dict[annot[0]].update({annot[1:]: gid})  #(start, stop, strd)

    # Compute padding dict from log info (circular, for the sake of reusing
    # bedgraph_loader() code)
    pad_dict = {}
    xvals = {}
    for gid in gene_ids:
        begin, end = fit_parse.log[gid]['fit_range']
        _, start, stop, _ = fit_parse.annotations[gid]
        gene_len = abs(stop - start)
        pad_dict[gid] = (abs(begin), abs(end - gene_len))

    # Reads data. Format: {'gene_id': (preads, nreads), ...}
    reads_dict = dp.bedgraph_loader(
        bgp_file, 
        bgn_file, 
        annot_dict, 
        pad_dict, 
        chr_order=chr_order
    )

    # Format and consolidate all the results for return
    results = OrderedDict()
    for gid in gene_ids:
        xvals = np.array(range(*fit_parse.log[gid]['fit_range']))
        strand = fit_parse.annotations[gid][3]
        if strand == 1:
            start = fit_parse.annotations[gid][1]
        else:
            xvals = invert(xvals, 0)
            start = fit_parse.annotations[gid][2]

        preads = [i-start for i in dp.reads_d2l(reads_dict[gid][0])]
        nreads = [i-start for i in dp.reads_d2l(reads_dict[gid][1])]
        model_params = {p:v[0] for p, v in fit_parse.fits[gid].items()}

        # Round w_b and extend w_a
        wb_update = np.around(1.0 - sum(model_params['w'][0:3]), decimals=2)
        model_params['w'] = [*model_params['w'][0:3], wb_update]
        if len(model_params['w_a']) == 2:
            w_a = model_params['w_a']
            model_params['w_a'] = [w_a[0], 0, 0, w_a[1]]

        results[gid] = (xvals, preads, nreads, strand, model_params)

    return results


def pdf_generator(
    mL = None,
    sL = None,
    tI = None,
    mT = None,
    sT = None,
    w = None,
    mL_a = None,
    sL_a = None,
    tI_a = None,
    w_a = None,
    strand = None,
    xvals = None,
    data = None
):
    '''
    This function is used in LIET_ax() in plotting library for generating the 
    model pdf's from the set of model parameters. Meant to be used downstream 
    of results_loader() and calls gene_model() function from data sim lib. 
    '''
    # Scaling method for relative scaling of strands
    try:
        Np = len(data[0])
        Nn = len(data[1])

        frac_p = Np / (Np + Nn)
        frac_n = Nn / (Np + Nn)
    except:
        print("WARNING: Using unity scaling for both strands. No strand bias.")
        frac_p = 1.0
        frac_n = 1.0

    if strand == '+' or strand == 1:
        
        # If gene on pos strand, must invert antisense strand position
        mL_a = invert(mL_a, 0)

        plot_params = dict(
            mu0_p = mL, 
            sig0_p = sL, 
            tau0_p = tI, 
            mu1_p = mT, 
            sig1_p = sT,
            mu0_n = mL_a, 
            sig0_n = sL_a, 
            tau0_n = tI_a, 
            mu1_n = None, 
            sig1_n = None,
            w_p = w,
            w_n = w_a,
            N_p = 1000,
            N_n = 1000,
            rvs = False, 
            pdf = True
        )

    else:
        # If gene on neg strand, must invert sense strand positions
        print(f"ML, MT (preflip): {mL}, {mT}")
        mL = invert(mL, 0)
        mT = invert(mT, 0)

        plot_params = dict(
            mu0_p = mL_a, 
            sig0_p = sL_a, 
            tau0_p = tI_a, 
            mu1_p = None, 
            sig1_p = None,
            mu0_n = mL, 
            sig0_n = sL, 
            tau0_n = tI, 
            mu1_n = mT, 
            sig1_n = sT,
            w_p = w_a,
            w_n = w,
            N_p = 1000,
            N_n = 1000,
            rvs = False, 
            pdf = True
        )
        print(f"ML, MT, ML_A: {mL}, {mT}, {mL_a}")

    # Generate pdfs for fitting results
    print(f"xvals range: {min(xvals)}, {max(xvals)}")
    pdf_p, pdf_n = gene_model(xvals, **plot_params)

    return pdf_p, pdf_n, frac_p, frac_n


def hist_generator(
    data,
    fractions,
    nbins = 'auto',
):
    '''
    Generates historgram and scaled heights (including signed orientation) for 
    both strands of data. Fractions tuple is used to scale the data on each 
    strand according to the strand bias. Intended for use with pdf_generator() 
    in LIET_ax() function for creating data and model plots.
    '''
    data_p, data_n = data
    frac_p, frac_n = fractions
    print(f"P DATA RANGE: {min(data_p)}, {max(data_p)}")
    print(f"N DATA RANGE: {min(data_n)}, {max(data_n)}")

    bmin = min(min(data_p), min(data_n))
    bmax = max(max(data_p), max(data_n))
    if nbins == "auto":
        bins = "auto"
    else:
        bins = np.linspace(bmin, bmax, nbins)

    # Make positive oriented histogram for postive strand
    hist_p = np.histogram(data_p, bins=bins, density=True)
    height_p = +hist_p[0] * frac_p

    # Make negative oriented histogram for negative strand
    hist_n = np.histogram(data_n, bins=bins, density=True)
    height_n = -hist_n[0] * frac_n

    return hist_p, hist_n, height_p, height_n
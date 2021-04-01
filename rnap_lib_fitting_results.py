import numpy as np
import scipy as sp
import pymc3 as pm
import time

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
    approx = vi.fit(iterations, progressbar=True, **fit_dict)

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
    # Sample approximation
    posterior_samples = post_approx.sample(N)

    # Filter out transformed variable names
    params = [e for e in posterior_samples.varnames if e[-2:] != '__']

    # Mode computing function that uses guassian kde to smooth empirical dist
    def kde_mode(samples, tol=1e-2):
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

#        try:
        samps = posterior_samples[p][:]

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

#       except:
#            print(f"WARNING: Can't compute stat for {p}.")
#            continue

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


def results_format(gene_id, post_stats, stat='mean', decimals=2):
    '''
    Parameters
    ----------
    gene_id : str
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
    params = ['mL', 'sL', 'tI', 'mT', 'sT', 'w', 'mL_a', 'sL_a', 'tI_a', 'w_a']
    fields = list([gene_id])

    for p in params:
        
        pvals = post_stats[p]

        pval = np.around(pvals[stat], decimals=decimals)
        pstd = np.around(pvals['stdev'], decimals=decimals)

        pstring = f"{p}={pval}:{pstd}"
        fields.append(pstring)

    res = "\t".join(fields) + "\n"
    return res

#    for pname, pvals in post_stats.items():
        
#        pval = np.around(pvals[stat], decimals=decimals)
#        pstd = np.around(pvals['stdev'], decimals=decimals)

#        pstring = f"{pname}={pval}:{pstd}"
#        fields.append(pstring)

#    res = "\t".join(fields) + "\n"

#    return res


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
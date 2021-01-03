import numpy as np
import scipy as sp
import pymc3 as pm

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
    approx = vi.fit(iterations, **fit_dict)

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
    mean=True,
    mode=True,
    median=False,
    stdev=True,
    skew=False
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

    mean : bool
        Indicate whether or not to return the mean of the posteriors.
    
    mode : bool
        Indicate whether or not to return the mode of the posteriors.

    median : bool
        Indicate whether or not to return the median of the posteriors.

    stdev : bool
        Indicate whether or not to return the standard deviation of the 
        posteriors.
    
    skew : bool
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

                mode = max(zip(x,y), key = lambda x : x[1])[0]
                modes = np.append(modes, mode)
        
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

        if mean:
            mean = np.mean(samps, axis=0)
        else:
            mean = None

        if median:
            median = np.median(samps, axis=0)
        else:
            median = None

        if stdev:
            stdev = sp.stats.tstd(samps, axis=0)
        else:
            stdev = None

        if mode:
            
            mode = kde_mode(samps)
        else:
            mode = None

        if skew:
            skew = sp.stats.skew(samps, axis=0)
            skewtest = sp.stats.skewtest(samps, axis=0).pvalue
        else:
            skew = None
            skewtest = None

        post_stats[p] = {
            'mean': mean,
            'median': median,
            'stdev': stdev,
            'mode': mode,
            'skew': skew,
            'skewtest': skewtest
        }

#        except:
#            print(f"WARNING: Posterior for {p} not present.")
#            continue

    return post_stats
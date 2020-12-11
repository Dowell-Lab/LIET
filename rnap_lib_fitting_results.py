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

    method : string
        ...

    optimizer : string
        ...

    learning_rate : float
        ...
    
    start : dictionary
        ...

    iterations : integer
        ...

    tolerance : float
        ...

    param_tracker : boolian
        ...


    Returns
    -------
    fit : dictionary
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

# def posterior_stats(
#     posterior_samples,
#     mean=True,
#     mode=True,
#     median=False,
#     stdev=True,
#     skew=False
# ):
#     '''
#     Compute specified stats for the posterior samples.
#     '''

#     # Mode computing function that uses guassian kde to smooth empirical dist
#     def kde_mode(samples, tol=1e-2):
#         '''
#         Computes the mode of the distribution after applying a guassian kde
#         '''
#         smin = min(samples)
#         smax = max(samples)
#         N = int((smax - smin)/tol)
        
#         samp_kde = sp.stats.gaussian_kde(samples)
#         x = np.linspace(smin, smax, N)
#         y = samp_kde.pdf(x)

#         mode = max(zip(x,y), key = lambda x : x[1])[0]

#         return mode


#     for p in self._pmap.keys():

#         try:
#             # Pad the w_b weight with zeros, if it was not part of fit
#             if p == 'w':
#                 samps = self.results['posteriors'][p][:]
#                 zeros = np.zeros((len(samps), 1))
#                 samps = np.concatenate((samps, zeros), axis=1)

#             else:
#                 samps = self.results['posteriors'][p][:]

#             # Compute the stats for parameter p
#             mean = np.mean(samps, axis=0)
#             median = np.median(samps, axis=0)
#             std = sp.stats.tstd(samps, axis=0)
# #                mode = sp.stats.mode(samps, axis=0)[0][0]
#             mode = kde_mode(samps)
#             skew = sp.stats.skew(samps, axis=0)
#             skewtest = sp.stats.skewtest(samps, axis=0).pvalue

#             self.results[p] = {
#                 'mean': mean,
#                 'median': median,
#                 'std': std,
#                 'mode': mode,
#                 'skew': skew,
#                 'skewtest': skewtest
#             }

#         except:
#             print(f"WARNING: Posterior for {p} not present.")
#             continue
        
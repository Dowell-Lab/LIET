import numpy as np
import pymc3 as pm

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
    Function that performs fitting using the VI framework.
    '''
    fit = {
        'vi': None,
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
    vi.fit(iterations, **fit_dict)

    # Add results to output dict
    fit['vi'] = vi
    if tracker:
        fit['tracker'] = tracker

    return fit

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
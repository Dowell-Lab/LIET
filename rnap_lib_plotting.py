import matplotlib.pyplot as plt

import rnap_lib_data_sim as ds


def convergence_plot(
        inference=None, 
        tracker=None, 
        fig_size=(16, 9),
        save=None,
        dpi=600
    ):

    '''
    Plotting for various convergence metrics. Namely, mean and stdev trackers 
    for mean-field estimations for model parameters as well as the objective 
    function ELBO. Can plot one or the other or both.

    Parameters
    ----------
    inference : pymc3 VI object
        Inference associated with fitting. This is the ADVI or FullRankADVI.
        Must have .hist object.

    tracker : pymc3 Tracker callback
        The tracker must contain 'mean' and 'std' keys.

    fig_size : numeric tuple
        Tuple specifying figure size for matplotlib figure. Default: (16, 9).

    save : string
        Full path (including filename) to save location for figure.

    dpi : integer
        dpi provided to .savefig() for figure file. Default: 600


    Returns
    -------
    fig : Matplotlib figure object

    '''

    fig = plt.figure(figsize=fig_size)

    if inference and tracker:

        mu_ax = fig.add_subplot(221)
        std_ax = fig.add_subplot(222)
        hist_ax = fig.add_subplot(212)

        mu_ax.plot(tracker['mean'])
        mu_ax.set_title('Mean tracker')
        std_ax.plot(tracker['std'])
        std_ax.set_title('Std tracker')
        hist_ax.plot(inference.hist)
        hist_ax.set_title('Negative ELBO tracker')
        hist_ax.set_xlabel('Iterations')

    elif inference and not tracker:
        
        hist_ax = fig.add_subplot(111)
        hist_ax.plot(inference.hist)
        hist_ax.set_title('Negative ELBO')
        hist_ax.set_xlabel('Iterations')

    elif not inference and tracker:

        mu_ax = fig.add_subplot(211)
        std_ax = fig.add_subplot(212)

        mu_ax.plot(tracker['mean'])
        mu_ax.set_title('Mean tracker')
        std_ax.plot(tracker['std'])
        std_ax.set_title('Std tracker')
        std_ax.set_xlabel('Iterations')
    else:
        return 0

    if isinstance(save, str):
        fig.savefig(save, bbox_inches='tight', dpi=dpi)

    return fig



#==============================================================================
def LIET_plot(
        liet_class, 
        stat='mean',
        bins='auto',
        data=True,
        sense=True,
        antisense=False,
        fig_size=(10, 7),
        save=None,
        dpi=600):
    '''
    Plot the model pdf that results from fitting. `stat` specifies which 
    value should be used for the parameter value. Option is either 'mean' 
    or 'mode'.

    Parameters
    ----------

    fig_size : numeric tuple
        Tuple specifying figure size for matplotlib figure. Default: (16, 9).

    save : string
        Full path (including filename) to save location for figure.

    dpi : integer
        dpi provided to .savefig() for figure file. Default: 600


    Returns
    -------
    fig : Matplotlib figure object
    '''
    # Check that parameter stats exist.
    if liet_class.results['mL']:

        fig = plt.figure(figsize=fig_size)

        mL = liet_class.results['mL'][stat]
        sL = liet_class.results['sL'][stat]
        tI = liet_class.results['tI'][stat]
        mT = liet_class.results['mT'][stat]
        sT = liet_class.results['sT'][stat]
        w = liet_class.results['w'][stat]
        if len(w) == 3:
            w.extend([0])

        pdf = ds.gene_model(
            liet_class.data['coord'],
            mu0 = mL,
            sig0 = sL,
            tau0 = tI,
            mu1 = mT,
            sig1 = sT,
            weights = w,
            rvs = False
        )

        plt.plot(liet_class.data['coord'], pdf)
        
    else:
        print(
            "WARNING: Optimal parameter values not in `self.results`. "
            "Must first run fitting routine and posterior_summary()."
        )
        return 0

        if data:
            if self.data['annot']['strand'] == 1:
                plt.hist(self.data['pos_reads'], density=True, bins=bins)
                plt.legend(['LIET fit', 'Data'])
            else:
                plt.hist(self.data['neg_reads'], density=True, bins=bins)
                plt.legend(['LIET fit', 'Data'])



    if isinstance(save, str):
        fig.savefig(save, bbox_inches='tight', dpi=dpi)

    return fig
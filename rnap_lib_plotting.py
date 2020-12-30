import numpy as np

import matplotlib.pyplot as plt

import rnap_lib_data_sim as ds

#==============================================================================

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

        rmap = inference.approx.groups[0].bij.rmap
        labels = rmap(np.array(range(len(tracker['mean'][0]))))
        labels = [k for k, v in sorted(labels.items(), key=lambda x : x[1])]
        labels = [e.split('_')[0] for e in labels]                          ## This doesn't quite work because of parameters with underscores, like 'w_a'

        std_ax.legend(
            labels, 
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5)
        )

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
        antisense=True,
        shifted = True,
        fig_size=(10, 7),
        save=None,
        dpi=600):
    '''
    Plot the model pdf that results from fitting. `stat` specifies which 
    value should be used for the parameter value. Option is either 'mean' 
    or 'mode'.

    Parameters
    ----------
    liet_class : LIET class object
        LIET class object which has strand data and fit results. The `.results`
         must contain values for each parameter for the specified statistic.

    stat : str
        String specifying which summary statistic to for each model parameter. 
        To be used in generating the best fit pdf from `.gene_model()`.
        Available options: 'mean', 'median', 'mode'. Default: 'mean'

    bins : str
        String specifying the bins or binning method, which is passed to 
        `np.hist()` method. Default: 'auto'

    data : bool
        Boolean specifying whether or not to plot the data along with the best 
        fit model. Read data is sourced from `liet_class.data` dict, and the 
        genomic coordinates given by `liet_class.data['coord']`.

    sense : bool
        Boolean specifying whether or not to plot sense strand data and model. 
        Default: True

    antisense : bool
        Boolean specifying whether or not to plot antisense strand data and 
        model. Default: True
    
    shifted : bool
        Boolean specifying whether or not strand data (and the model results) 
        had been positively rectified for fitting, which is specified in the 
        LIET class object (`liet_class.data['shift']`). Default: True

    fig_size : numeric tuple
        Tuple specifying figure size for matplotlib figure. Default: (16, 9).

    save : str
        Full path (including filename) to save location for figure. This will 
        overwrite any existing file.

    dpi : int
        dpi provided to .savefig() for figure file. Default: 600


    Returns
    -------
    fig : Matplotlib figure object
        Recommended that you set this function call equal to something, 
        otherwise it will display the figure twice.
    '''

    # Check that parameter stats exist.
    if liet_class.results['mL']:
        
        strand = liet_class.data['annot']['strand']
        xvals = liet_class.data['coord']
        results = liet_class.results

        if sense:
            mL = results['mL'][stat]
            sL = results['sL'][stat]
            tI = results['tI'][stat]
            mT = results['mT'][stat]
            sT = results['sT'][stat]
            w = results['w'][stat]
            if len(w) == 3:
                w.extend([0])

        else:
            mL, sL, tI, mT, sT, w = None, None, None, None, None, None
            
        if antisense:
            mL_a = results['mL_a'][stat]
            sL_a = results['sL_a'][stat]
            tI_a = results['tI_a'][stat]
            w_a = results['w_a'][stat]
            if len(w_a) == 3:
                w_a.extend([0])
        else:
            mL_a, sL_a, tI_a, w_a = None, None, None, None

        # Define pdf plotting parameters, depending on annotation strand
        if strand == '+' or strand == 1:
            
            # If gene on pos strand, must invert antisense strand position
            if shifted and antisense:
                mL_a = ds.invert(mL_a, 0)

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
            if shifted and sense:
                mL = ds.invert(mL, 0)
                mT = ds.invert(mT, 0)

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

        # Generate pdfs for fitting results
        pdf_p, pdf_n = ds.gene_model(xvals, **plot_params)

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        # Plot pdfs of gene dependant on <sense> and <antisense>
        if strand == '+' or strand == 1:
            if sense:
                ax.plot(xvals, pdf_p, color='C0')
            if antisense:
                ax.plot(xvals, -pdf_n, color='C3')
        else:
            if sense:
                ax.plot(xvals, -pdf_n, color='C3')
            if antisense:
                ax.plot(xvals, pdf_p, color='C0')

        # Plotting the read data
        if data:

            data_p = liet_class.data['pos_reads']
            data_n = liet_class.data['neg_reads']
            # Horizontal inversion if data was initially shifted
            if shifted:
                data_n = ds.invert(data_n, 0)
            
            # Make negative histogram for neg strand data, depending on annot
            if sense:
                if strand == '+' or strand == 1:
                    hist = np.histogram(data_p, bins=bins, density=True)
                    height = +hist[0]
                    col = 'C0'
                else:
                    hist = np.histogram(data_n, bins=bins, density=True)
                    height = -hist[0]
                    col = 'C3'

                loc = hist[1][:-1]
                width = hist[1][1] - hist[1][0]
                ax.bar(loc, height, width, alpha=0.3, align='edge', color=col)

            if antisense:
                if strand == '+' or strand == 1:
                    hist = np.histogram(data_n, bins=bins, density=True)
                    height = -hist[0]
                    col = 'C3'
                else:
                    hist = np.histogram(data_p, bins=bins, density=True)
                    height = +hist[0]
                    col = 'C0'

                loc = hist[1][:-1]
                width = hist[1][1] - hist[1][0]
                ax.bar(loc, height, width, alpha=0.3, align='edge', color=col)

        # Add legend, labels, and reference line
        ax.axvline(
            x=0, ymin=0, ymax=1, 
            linestyle='--', 
            linewidth=0.5, 
            color='k'
        )
        annot = liet_class.data['annot']
        txt = f"{annot['chrom']}:{annot['start']}"
        xlim = ax.get_xlim()
        txt_pos = -1*xlim[0] / (xlim[1] - xlim[0]) + 0.003
        ax.text(
            txt_pos, 0.02, txt,
            horizontalalignment='left', 
            verticalalignment='bottom', 
            transform=ax.transAxes
        )

        ax.set_xlabel(
            f"Genomic position (relative to gene TSS: {annot['gene_id']})"
        )
        ax.set_ylabel('Fractional probability')

    else:
        print(
            "WARNING: Optimal parameter values not in `self.results`. "
            "Must first run fitting routine and posterior_summary()."
        )
        return 0

    if isinstance(save, str):
        ax.savefig(save, bbox_inches='tight', dpi=dpi)

    return fig
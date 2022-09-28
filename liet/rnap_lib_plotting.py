import numpy as np

import matplotlib.pyplot as plt

import rnap_lib_data_sim as ds
import rnap_lib_data_proc as dp
from liet_res_class import FitParse

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

    if inference != None and tracker != None:

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

#        rmap = inference.approx.groups[0].bij.rmap
#        labels = rmap(np.array(range(len(tracker['mean'][0]))))
#        print(f"labels: {labels}")
#        labels = [k for k, v in sorted(labels.items(), key=lambda x : x[1])]
#        print(f"labels 2: {labels}")
#        labels = [e.split('_')[0] for e in labels]                          ## NEED TO FIX: This doesn't quite work because of parameters with underscores, like 'w_a'

#        std_ax.legend(
#            labels, 
#            loc='center left', 
#            bbox_to_anchor=(1.02, 0.5)
#        )

    elif inference != None and tracker == None:
        
        hist_ax = fig.add_subplot(111)
        hist_ax.plot(inference.hist)
        hist_ax.set_title('Negative ELBO')
        hist_ax.set_xlabel('Iterations')

    elif inference == None and tracker != None:

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
        xlim=None,
        ylim=None,
        save=False,
        dpi=600):
    '''
    Plot the model pdf that results from fitting. 'stat' specifies which 
    value should be used for the parameter value. Option is either 'mean' 
    or 'mode'. Requires input of fully populated LIET class instance.

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

    # Scaling method for relative scaling of strands
    try:
        Np = len(liet_class.data['pos_reads'])
        Nn = len(liet_class.data['neg_reads'])

        frac_p = Np / (Np + Nn)
        frac_n = Nn / (Np + Nn)
    except:
        frac_p = 1.0
        frac_n = 1.0

    # Check that parameter stats exist.
    if liet_class.results['mL']:
        
        strand = liet_class.data['annot']['strand']
        if strand == '+' or strand == 1:
            xvals = liet_class.data['coord']
        else:
            xvals = np.flip(-1 * liet_class.data['coord'])

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
            # NOTE: I don't particularly like this handling. I should probably do this assigning further upstream (say in fr.posterior_stats())
            # I didn't realize the handling her had changed from when I used to rely on .posterior_summary() function contained in LIET class
            if 'mL_a' in results.keys():
                mL_a = results['mL_a'][stat]
            else:
                mL_a = results['mL'][stat]
            if 'sL_a' in results.keys():
                sL_a = results['sL_a'][stat]
            else:
                sL_a = results['sL'][stat]
            if 'tI_a' in results.keys():
                tI_a = results['tI_a'][stat]
            else:
                tI_a = results['tI'][stat]
            if 'w_a' in results.keys():
                w_a = results['w_a'][stat]
            else:
                w_a = results['w'][stat]
            # Have to do this so that the weights arrays are length 4
            if len(w_a) == 1:
                w_a.extend([0, 0, 0])
            elif len(w_a) == 2:
                w_a = [w_a[0], 0, 0, w_a[1]]
#            if len(w_a) == 3:
#                w_a.extend([0])
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
                ax.plot(xvals, pdf_p * frac_p, color='C0')
            if antisense:
                ax.plot(xvals, -pdf_n * frac_n, color='C3')
        else:
            if sense:
                ax.plot(xvals, -pdf_n * frac_n, color='C3')
            if antisense:
                ax.plot(xvals, pdf_p * frac_n, color='C0')

        # Plotting the read data
        if data:

            # TESTING BINS
#            bins = np.linspace(xvals[0], xvals[-1], 1000)

            data_p = liet_class.data['pos_reads']
            data_n = liet_class.data['neg_reads']

            # Horizontal inversion if data was initially shifted
            if shifted:
                data_n = ds.invert(data_n, 0)

            bmin = min(min(data_p), min(data_n))
            bmax = max(max(data_p), max(data_n))
            bins = np.linspace(bmin, bmax, 1000)

            # Make negative histogram for neg strand data, depending on annot
            if sense:
                if strand == '+' or strand == 1:
                    hist = np.histogram(data_p, bins=bins, density=True)
                    height = +hist[0] * frac_p
                    col = 'C0'
                else:
                    hist = np.histogram(data_n, bins=bins, density=True)
                    height = -hist[0] * frac_n
                    col = 'C3'

                loc = hist[1][:-1]
                width = hist[1][1] - hist[1][0]
                ax.bar(loc, height, width, alpha=0.2, align='edge', color=col)

            if antisense:
                if strand == '+' or strand == 1:
                    hist = np.histogram(data_n, bins=bins, density=True)
                    height = -hist[0] * frac_n
                    col = 'C3'
                else:
                    hist = np.histogram(data_p, bins=bins, density=True)
                    height = +hist[0] * frac_p
                    col = 'C0'

                loc = hist[1][:-1]
                width = hist[1][1] - hist[1][0]
                ax.bar(loc, height, width, alpha=0.2, align='edge', color=col)

        # Set plot limits
        if isinstance(xlim, list) and len(xlim) == 2:
            plt.xlim(xlim)
        if isinstance(ylim, list) and len(ylim) == 2:
            plt.ylim(ylim)

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

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=dpi)

    return fig


    #==========================================================================

def LIET_ax(
    ax,
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
    data = None,
    nbins = 'auto',
):

    '''
    Plotting function alternative to LIET_plot(). Plots onto matplotlib axis 
    object. Called from LIET_plot2() which I'm using to generate figures for 
    the paper.
    
    Parameters
    ----------


    Returns
    -------
    None
    '''
    # Scaling method for relative scaling of strands
    try:
        Np = len(data['pos_reads'])
        Nn = len(data['neg_reads'])

        frac_p = Np / (Np + Nn)
        frac_n = Nn / (Np + Nn)
    except:
        frac_p = 1.0
        frac_n = 1.0

    if strand == '+' or strand == 1:
        
        # If gene on pos strand, must invert antisense strand position
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

    # Plot pdfs of gene dependant on <sense> and <antisense>
    if strand == '+' or strand == 1:
        # Sense
        ax.plot(xvals, pdf_p * frac_p, color='C0')
        # Antisense
        ax.plot(xvals, -pdf_n * frac_n, color='C3')
    else:
        # Sense
        ax.plot(xvals, -pdf_n * frac_n, color='C3')
        # Antisense
        ax.plot(xvals, pdf_p * frac_n, color='C0')

    # Plotting the read data
    if data:
        data_p = data[0]
        data_n = data[1]

        # Horizontal inversion if data was initially shifted
#        data_n = ds.invert(data_n, 0)

        bmin = min(min(data_p), min(data_n))
        bmax = max(max(data_p), max(data_n))
        if nbins == "auto":
            bins = "auto"
        else:
            bins = np.linspace(bmin, bmax, nbins)

        # Make negative histogram for neg strand data, depending on annot
        if strand == '+' or strand == 1:
            hist = np.histogram(data_p, bins=bins, density=True)
            height = +hist[0] * frac_p
            col = 'C0'
        else:
            hist = np.histogram(data_n, bins=bins, density=True)
            height = -hist[0] * frac_n
            col = 'C3'

        loc = hist[1][:-1]
        width = hist[1][1] - hist[1][0]
        ax.bar(loc, height, width, alpha=0.2, align='edge', color=col)

        if strand == '+' or strand == 1:
            hist = np.histogram(data_n, bins=bins, density=True)
            height = -hist[0] * frac_n
            col = 'C3'
        else:
            hist = np.histogram(data_p, bins=bins, density=True)
            height = +hist[0] * frac_p
            col = 'C0'

        loc = hist[1][:-1]
        width = hist[1][1] - hist[1][0]
        ax.bar(loc, height, width, alpha=0.2, align='edge', color=col)



def LIET_plot2(
    ax,
    gene_id,
    config=None,
    result=None,
    log=None
):
    '''
    This function generates a LIET model fit plot in a matplotlib Axes object, 
    for a specified gene ID. It loads data and fit parameters from config, 
    bedGraph, and LIET fit results output file and then calls the LIET_ax() 
    function to perform the actual plotting.
    '''

    # Parse config and fit results files
    config_parse = dp.config_loader(config)
    fit_parse = FitParse(result, log_file=log)

    # Only need the input files from config
    bgp_file = config_parse['FILES']['BEDGRAPH_POS']
    bgn_file = config_parse['FILES']['BEDGRAPH_NEG']

    # Extract range and annot data for specific gene
    fit_range = fit_parse.log[gene_id]['fit_range']
    chromosome, start, stop, strand = fit_parse.annotations[gene_id]
    xvals = np.array(range(*fit_range))

    # Determine chromosome string order
    chr_order = dp.chrom_order_reader(bgp_file, bgn_file)

    # Extract read data for gene from bedGraphs
    with open(bgp_file, 'r') as bgp, open(bgn_file, 'r') as bgn:
        current_bgpl = ['chr1', 0, 0, 0]    # Initialize pos strand bg line
        current_bgnl = ['chr1', 0, 0, 0]    # Initialize neg strand bg line

        bgp_args = [
            bgp, 
            current_bgpl, 
            chromosome, 
            fit_range[0] + start,
            fit_range[1] + start,
            chr_order
        ]
        _, preads = dp.bgreads(*bgp_args)
        
        bgn_args = [
            bgn, 
            current_bgnl, 
            chromosome, 
            fit_range[0] + start,
            fit_range[1] + start,
            chr_order
        ]
        _, nreads = dp.bgreads(*bgn_args)

    # Convert read dicts to list and concatenate
    data = [dp.reads_d2l(preads), dp.reads_d2l(nreads)]

    # Shift data genomic coordinates to TSS
    data = [[i-start for i in data[0]], [i-start for i in data[1]]]

    # Extract parameter values from FitParse object
    model_params = {p:v[0] for p, v in fit_parse.fits[gene_id].items()}

    # Enforce that weights sum to 1.0 by adjusting wB
    wb_update = np.around(1.0 - sum(model_params['w'][0:3]), decimals=2)

    model_params['w'] = (*model_params['w'][0:3], wb_update)

    print(f"STRAND: {strand}")
    print(f"PARAMS: {model_params.items()}")
    # Generate plot on Axes object ax
    LIET_ax(
        ax,
        **model_params,
        strand = strand,
        xvals = xvals,
        data = data,
        nbins = 1000,
    )
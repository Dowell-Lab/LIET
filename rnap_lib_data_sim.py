import numpy as np
import scipy.stats as stats

## Coordinate shift function ==================================================

def inverter(x, mu):
    '''
    Inverts the xvals prior to input into the above functions to be used for 
    antisense strand. Inversion point <mu> is mL.
    '''
    x_invert = np.array([2*mu - e for e in x])
    return x_invert



## Model PDF components =======================================================

def load_initiation_pdf(x, m, s, t):
    pdf = stats.exponnorm.pdf(x, t/s, m, s)
    return pdf


def elongation_pdf(x, m0, s0, t0, m1, s1):
    unscaled = (stats.exponnorm.cdf(x, t0/s0, m0, s0) 
        * stats.norm.sf(x, m1, s1))
    
    xmin = int(min(m0 - 10*s0, m1 - 10*s1))
    xmax = int(max(m0 + 10*np.sqrt(s0**2 + t0**2), m1 + 10*s1))
    xfull = np.array(range(xmin, xmax))
    
    norm_factor = sum(stats.exponnorm.cdf(xfull, t0/s0, m0, s0) 
        * stats.norm.sf(xfull, m1, s1))
    pdf = unscaled/norm_factor
    return pdf


def termination_pdf(x, m, s):
    pdf = stats.norm.pdf(x, m, s)
    return pdf


def background_pdf(x):
    length = len(x)
    pdf = np.ones(length) * (1/length)
    return pdf



## Model RVS components =======================================================

def load_initiation_rvs(x, m, s, t, size=1000, seed=42):
    samples = stats.exponnorm.rvs(t/s, m, s, size=size, random_state=seed)
    return samples


def elongation_rvs(x, m0, s0, t0, m1, s1, size=1000, seed=42):
    
    pdf = elongation_pdf(x, m0=m0, s0=s0, t0=t0, m1=m1, s1=s1)
    # Adjust pdf to sum to 1.0
    residue = 1.0 - sum(pdf)
    pdf[-1] = pdf[-1] + abs(residue)
        
    np.random.seed(seed=seed)
    samples = np.random.choice(x, size=size, replace=True, p=pdf)
    return samples


def termination_rvs(x, m, s, size=1000, seed=42):
    samples = stats.norm.rvs(m, s, size=size, random_state=seed)
    return samples


def background_rvs(x, size=10, seed=42):
    np.random.seed(seed=seed)
    samples = np.random.choice(x, size=size, replace=True)
    return samples



## Full model PDF and RVS =====================================================

def gene_model(
    xvals, 
    mu0=None, 
    sig0=None, 
    tau0=None, 
    mu1=None, 
    sig1=None, 
    mu_a=None,
    sig_a=None,
    tau_a=None,
    weights=[0.7, 0.2, 0.09, 0.01], 
    bias = None,
    N=5000, 
    seed=42, 
    rvs=False,
    pdf=True,
):
    '''
    Parameters
    ----------
    xvals : numpy array
        Genomic coordinates on which to evaluate the model. Array of integers.

    mu0, sig0, tau0 : float kwargs
        Model parameters specifying the sense-strand Loading/Initiation EMG
    
    mu1, sig1 : float kwargs
        Model parameters specifying the sense-strand Termination guassian
    
    mu_a, sig_a, tau_a : float kwargs
        Model parameters specifying the antisense-strand Loading/Initiation EMG
        NOTE: if <bias = None>, anti-sense pdf and rvs wont be generated.

    weights : list (length == 4)
        Weights specifying Loading/Initiation, Elongation, Termination and 
        Background, in that order: [LI, E, T, B]. Must sum to 1. Background 
        weight is used for anti-sense strand as well.

    bias : float
        The "strand bias," e.g. the fraction of reads that come from the sense 
        strand. Must be between 0.0 and 1.0, or if None, anti-sense components 
        are not computed (Default: None).
    
    N : integer
        Total number of reads to generate, distributed across all components, 
        according to <weights> and <bias> parameters.

    seed : number
        Random seed for reproducibility of rvs samples.

    rvs : bool
        Indicates whether or not to return rvs

    pdf : bool
        Indicates whether or not to return pdf


    Returns
    -------
    pdf, pdf_a, rvs, rvs_a : numpy arrays
        ...
    '''
    # Unpack weights
    w5, we, w3, wb = weights

    # Generate PDF
    if pdf:
        li_pdf = load_initiation_pdf(
            xvals, 
            m=mu0, 
            s=sig0, 
            t=tau0
        )
        e_pdf = elongation_pdf(
            xvals, 
            m0=mu0, 
            s0=sig0, 
            t0=tau0, 
            m1=mu1, 
            s1=sig1
        )
        t_pdf = termination_pdf(xvals, m=mu1, s=sig1)        
        back_pdf = background_pdf(xvals)

        pdf_return = bias*(w5*li_pdf + we*e_pdf + w3*t_pdf + wb*back_pdf)

        if bias:
            li_a_pdf = load_initiation_pdf(
                inverter(xvals, mu_a), 
                m=mu_a, 
                s=sig_a, 
                t=tau_a
            )
            back_a = background_pdf(inverter(xvals, mu_a))

            pdf_a_return = (1-bias)*((1-wb)*li_a_pdf + wb*back_a)

    # Generate RV samples
    if rvs:

        # 'bias' is the fraction of reads from the sense strand
        if bias == None:
            N5 = int(w5 * N)
            NE = int(we * N)
            N3 = int(w3 * N)
            Nb = int(wb * N)
        else:
            N5 = int(w5 * N * bias)
            NE = int(we * N * bias)
            N3 = int(w3 * N * bias)
            Nb = int(wb * N * bias)
            # Number of reads on the antisense strand
            Nb_a = int(wb * N * (1-bias))
            N5_a = int(N - N5 - NE - N3 - Nb - Nb_a)

            li_rvs_a = load_initiation_rvs(
                xvals,
                mu_a,
                sig_a,
                tau_a,
                size=N5_a,
                seed=seed
            )
            li_rvs_a = [-e + 2*mu_a for e in li_rvs_a]  # flip about mu_a
            background_a = background_rvs(xvals, size=Nb_a, seed=seed)

            # Concatenate all antisense-strand model components
            rvs_a_return = np.concatenate([li_rvs_a, background_a])
            rvs_a_return = np.array(np.around(rvs_a_return, decimals=0), 
                dtype='int32')
            rvs_a_return = np.sort(rvs_a_return)

        li_rvs = load_initiation_rvs(
            xvals, 
            mu0, 
            sig0, 
            tau0, 
            size=N5, 
            seed=seed
        )
        e_rvs = elongation_rvs(
            xvals, 
            mu0, 
            sig0, 
            tau0, 
            mu1, 
            sig1, 
            size=NE, 
            seed=seed
        )
        t_rvs = termination_rvs(xvals, mu1, sig1, size=N3, seed=seed)
        
        background = background_rvs(xvals, size=Nb, seed=seed)
        
        # Concatenate together all sense-strand model components
        rvs_return = np.concatenate([li_rvs, e_rvs, t_rvs, background])
        rvs_return = np.array(np.around(rvs_return, decimals=0), dtype='int32')
        rvs_return = np.sort(rvs_return)

    # Returns
    if bias == None:
        if pdf and rvs:
            return pdf_return, rvs_return
        elif pdf and not rvs:
            return pdf_return
        else:
            return rvs_return
    else:
        if pdf and rvs:
            return pdf_return, pdf_a_return, rvs_return, rvs_a_return
        elif pdf and not rvs:
            return pdf_return, pdf_a_return
        else:
            return rvs_return, rvs_a_return
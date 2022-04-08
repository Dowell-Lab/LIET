import numpy as np
import scipy.stats as stats

## Coordinate shift function ==================================================

def invert(x, mu):
    '''
    Inverts the <x> (int, float, or iterable) prior to input into the above 
    functions to be used for antisense strand. Inversion point <mu>.
    '''
    if isinstance(x, (float, int)):
        return 2*mu - x
    else:
        return np.array([2*mu - e for e in x])



## Model PDF components =======================================================

def load_initiation_pdf(x, m, s, t):
    pdf = stats.exponnorm.pdf(x, t/s, m, s)
    return pdf

# Note elongation_pdf must be inverted internally because it has 2 inversion 
# points --- <m0> and <m1>.
def elongation_pdf(x, m0, s0, t0, m1, s1):

    if m0 <= m1:
        cdf = stats.exponnorm.cdf(x, t0/s0, m0, s0)
        sf = stats.norm.sf(x, m1, s1)
        unscaled = np.nan_to_num(cdf * sf)
        #unscaled = (stats.exponnorm.cdf(x, t0/s0, m0, s0) 
        #    * stats.norm.sf(x, m1, s1))
    else:
        cdf = stats.exponnorm.cdf(invert(x, m0), t0/s0, m0, s0)
        sf = stats.norm.sf(invert(x, m1), m1, s1)
        unscaled = np.nan_to_num(cdf * sf)
        #unscaled = (stats.exponnorm.cdf(invert(x, m0), t0/s0, m0, s0) 
        #    * stats.norm.sf(invert(x, m1), m1, s1))
    
    xmin = int(min(m0 - 10*s0, m1 - 10*s1))
    xmax = int(max(m0 + 10*np.sqrt(s0**2 + t0**2), m1 + 10*s1))
    xfull = np.array(range(xmin, xmax))
    
    if m0 <= m1:
        cdf = stats.exponnorm.cdf(xfull, t0/s0, m0, s0)
        sf = stats.norm.sf(xfull, m1, s1)
        norm_factor = sum(np.nan_to_num(cdf * sf))
        #norm_factor = sum(stats.exponnorm.cdf(xfull, t0/s0, m0, s0) 
        #    * stats.norm.sf(xfull, m1, s1))
    else:
        cdf = stats.exponnorm.cdf(invert(xfull, m0), t0/s0, m0, s0)
        sf = stats.norm.sf(invert(xfull, m1), m1, s1)
        norm_factor = sum(np.nan_to_num(cdf * sf))
        #norm_factor = sum(stats.exponnorm.cdf(invert(xfull, m0), t0/s0, m0, s0) 
        #    * stats.norm.sf(invert(xfull, m1), m1, s1))

    pdf = np.nan_to_num(unscaled/norm_factor)
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
    
    # Adjust <x> so that it encompasses full normalization range (xfull)
    xmin = int(min(m0 - 10*s0, m1 - 10*s1))
    xmax = int(max(m0 + 10*np.sqrt(s0**2 + t0**2), m1 + 10*s1))
    xfull = np.array(range(xmin, xmax))

    pdf = elongation_pdf(xfull, m0=m0, s0=s0, t0=t0, m1=m1, s1=s1)

    # Adjust pdf to sum to 1.0 (residue from finite normalization integration)
    residue = 1.0 - sum(pdf)
    pdf[-1] = pdf[-1] + abs(residue)
        
    np.random.seed(seed=seed)
    samples = np.random.choice(xfull, size=size, replace=True, p=pdf)
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
    mu0_p=None, 
    sig0_p=None, 
    tau0_p=None, 
    mu1_p=None, 
    sig1_p=None, 
    mu0_n=None, 
    sig0_n=None, 
    tau0_n=None, 
    mu1_n=None, 
    sig1_n=None, 
    w_p=[0.7, 0.2, 0.09, 0.01],
    w_n=[0.99, 0, 0, 0.01],
    N_p=1000,
    N_n=1000,
    seed=42, 
    rvs=False,
    pdf=True,
):
    '''
    Parameters
    ----------
    xvals : numpy array
        Genomic coordinates on which to evaluate the model. Array of integers.

    mu0_p, sig0_p, tau0_p : float kwargs
        Model parameters specifying the positive-strand Loading/Initiation EMG
    
    mu1_p, sig1_p : float kwargs
        Model parameters specifying the positive-strand Termination gaussian
    
    mu0_n, sig0_n, tau0_n : float kwargs
        Model parameters specifying the negative-strand Loading/Initiation EMG
    
    mu1_n, sig1_n : float kwargs
        Model parameters specifying the negative-strand Termination gaussian

    w_p : list (length == 4)
        Weights specifying Loading/Initiation, Elongation, Termination and 
        Background on the positive strand. In order: [LI, E, T, B]. Weights 
        must sum to 1.
    
    w_n : list (length == 4)
        Weights specifying Loading/Initiation, Elongation, Termination and 
        Background on the negative strand. In order: [LI, E, T, B]. Weights 
        must sum to 1. NOTE: if `w_n = None`, no pdf or rvs will be generated
        for the negative strand.
    
    N_p : integer
        Total number of reads to generate from the positive strand pdf, 
        distributed according to <w_p>.

    seed : number
        Random seed for reproducibility of rvs samples.

    rvs : bool
        Indicates whether or not to return rvs

    pdf : bool
        Indicates whether or not to return pdf


    Returns
    -------
    pdf_p, pdf_n, rvs_p, rvs_n : numpy arrays
        Returned probability density functions and random variable samples for 
        the two strands. Number of arrays returned depends on <rvs> and <pdf> 
        boolian parameters.
    '''

    # Check for correct orientation of loading and termination positions
    if mu1_p != None and mu0_p != None:
        if mu1_p < mu0_p:
            raise ValueError('Loading position parameter <mu0_p> must be '
                'upstream of termination position <mu1_p>.')
    if mu1_n != None and mu0_n != None:
        if mu0_n < mu1_n:
            raise ValueError('Loading position parameter <mu0_n> must be '
                'upstream of termination position <mu1_n>.')

    # Recast weights
    w_p = np.array(w_p)
    w_n = np.array(w_n)

    # Check and unpack weights
    if w_p.all() != None:
        if len(w_p) == 2:
            w_n = np.array([w_n[0], 0.0, 0.0, w_n[1]])
        if round(sum(w_p), 5) == 1.0:
            wLI_p, wE_p, wT_p, wB_p = w_p
        else:
            raise ValueError('Weights parameter <w_p> must sum to 1.0')
    if w_n.all() != None:
        if len(w_n) == 2:
            w_n = np.array([w_n[0], 0.0, 0.0, w_n[1]])
        if round(sum(w_n), 5) == 1.0:
            wLI_n, wE_n, wT_n, wB_n = w_n
        else:
            raise ValueError('Weights parameter <w_n> must sum to 1.0')

    # Generate PDF(s)
    if pdf:
        # Positive strand pdf
        if w_p.all() != None:

            pdf_p = np.zeros(len(xvals))

            if wLI_p != 0.0:
                li_pdf_p = load_initiation_pdf(
                    xvals, 
                    m=mu0_p, 
                    s=sig0_p, 
                    t=tau0_p
                )
                pdf_p += wLI_p * li_pdf_p 

            if wE_p != 0.0:
                e_pdf_p = elongation_pdf(
                    xvals, 
                    m0=mu0_p, 
                    s0=sig0_p, 
                    t0=tau0_p, 
                    m1=mu1_p, 
                    s1=sig1_p
                )
                pdf_p += wE_p * e_pdf_p

            if wT_p != 0.0:
                t_pdf_p = termination_pdf(xvals, m=mu1_p, s=sig1_p)
                pdf_p += wT_p * t_pdf_p

            if wB_p != 0.0:
                back_pdf_p = background_pdf(xvals)
                pdf_p += wB_p * back_pdf_p
        
        else:
            pdf_p = np.array([])

        # Negative strand pdf
        if w_n.all() != None:

            pdf_n = np.zeros(len(xvals))

            if wLI_n != 0.0:
                li_pdf_n = load_initiation_pdf(
                    invert(xvals, mu0_n),
                    m=mu0_n, 
                    s=sig0_n, 
                    t=tau0_n
                )
                pdf_n += wLI_n * li_pdf_n

            if wE_n != 0.0:
            # Elongation pdf inverts internally
                e_pdf_n = elongation_pdf(
                    xvals, 
                    m0=mu0_n, 
                    s0=sig0_n, 
                    t0=tau0_n, 
                    m1=mu1_n, 
                    s1=sig1_n
                )
                pdf_n += wE_n * e_pdf_n

            if wT_n != 0.0:
                t_pdf_n = termination_pdf(
                    invert(xvals, mu1_n), 
                    m=mu1_n, 
                    s=sig1_n
                )  
                pdf_n += wT_n * t_pdf_n

            if wB_n != 0.0:
                back_pdf_n = background_pdf(invert(xvals, mu0_n))
                pdf_n += wB_n * back_pdf_n
        
        else:
            pdf_n = np.array([])


    # Generate RV samples
    if rvs:
        # Positive strand rvs
        if isinstance(N_p, int):
            Nli_p = int(wLI_p * N_p)
            Ne_p = int(wE_p * N_p)
            Nt_p = int(wT_p * N_p)
            Nb_p = int(wB_p * N_p)

            rvs_p = np.array([])

            if Nli_p != 0.0:
                li_rvs_p = load_initiation_rvs(
                    xvals, 
                    mu0_p, 
                    sig0_p, 
                    tau0_p, 
                    size=Nli_p, 
                    seed=seed
                )
                rvs_p = np.concatenate([rvs_p, li_rvs_p])
            
            if Ne_p != 0.0:
                # Elongation rvs invert internally
                e_rvs_p = elongation_rvs(
                    xvals, 
                    mu0_p, 
                    sig0_p, 
                    tau0_p, 
                    mu1_p, 
                    sig1_p, 
                    size=Ne_p, 
                    seed=seed
                )
                rvs_p = np.concatenate([rvs_p, e_rvs_p])
            
            if Nt_p != 0.0:
                t_rvs_p = termination_rvs(
                    xvals, 
                    mu1_p, 
                    sig1_p, 
                    size=Nt_p, 
                    seed=seed
                )
                rvs_p = np.concatenate([rvs_p, t_rvs_p])

            if Nb_p != 0:
                back_rvs_p = background_rvs(xvals, size=Nb_p, seed=seed)
                rvs_p = np.concatenate([rvs_p, back_rvs_p])
            
            rvs_p = np.array(np.sort(rvs_p), dtype='int32')
        
        else:
            rvs_p = np.array([])

        # Negative strand rvs
        if isinstance(N_n, int):
            Nli_n = int(wLI_n * N_n)
            Ne_n = int(wE_n * N_n)
            Nt_n = int(wT_n * N_n)
            Nb_n = int(wB_n * N_n)

            rvs_n = np.array([])

            if Nli_n != 0.0:
                li_rvs_n = load_initiation_rvs(
                    xvals, 
                    mu0_n, 
                    sig0_n, 
                    tau0_n, 
                    size=Nli_n, 
                    seed=seed
                )
                rvs_n = np.concatenate([rvs_n, invert(li_rvs_n, mu0_n)])

            if Ne_n != 0.0:
                # Elongation rvs invert internally
                e_rvs_n = elongation_rvs(
                    xvals, 
                    mu0_n, 
                    sig0_n, 
                    tau0_n, 
                    mu1_n, 
                    sig1_n, 
                    size=Ne_n, 
                    seed=seed
                )
                rvs_n = np.concatenate([rvs_n, e_rvs_n])

            if Nt_n != 0.0:
                t_rvs_n = termination_rvs(
                    xvals, 
                    mu1_n, 
                    sig1_n, 
                    size=Nt_n, 
                    seed=seed
                )
                rvs_n = np.concatenate([rvs_n, invert(t_rvs_n, mu1_n)])

            if Nb_n != 0.0:
                back_rvs_n = background_rvs(xvals, size=Nb_n, seed=seed)
                rvs_n = np.concatenate([rvs_n, back_rvs_n])

            rvs_n = np.array(np.sort(rvs_n), dtype='int32')
        
        else:
            rvs_n = np.array([])

    # Returns
    if pdf and rvs:
        return pdf_p, pdf_n, rvs_p, rvs_n
    elif pdf and not rvs:
        return pdf_p, pdf_n
    else:
        return rvs_p, rvs_n

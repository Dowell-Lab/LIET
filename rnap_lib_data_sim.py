import numpy as np
import scipy.stats as stats

### Model PDF components ##############################

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


### Model RVS components ##############################

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


### Full model PDF and RVS ####################################################

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
    pdf=True
):
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

        pdf_return = w5*li_pdf + we*e_pdf + w3*t_pdf + wb*back_pdf

# NOTE: I DON'T HAVE THE ANTI-SENSE STRAND INCORPORATED INTO THE PDF. IT IS NOT
# CLEAR HOW I WOULD FLIP THE GENOMIC COORDINATES YET.
#        if bias:
#            li_a_pdf = load_initiation_pdf(
#                xvals, 
#                m=mu_a, 
#                s=sig_a, 
#                t=tau_a
#            )

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
            return pdf_return, rvs_return, rvs_a_return
        elif pdf and not rvs:
            return pdf_return
        else:
            return rvs_return, rvs_a_return
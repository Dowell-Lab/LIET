#==============================================================================                                                                                                       
__author__ = 'Jacob T. Stanley'
__credits__ = ['Jacob T. Stanley', 'Robin D. Dowell']
__maintainer__ = 'Jacob T. Stanley'
__email__ = 'jacob.stanley@colorado.edu'                                                                                                       
#==============================================================================

import numpy as np
import scipy as sp
import pymc as pm
import rnap_lib_data_sim as ds

import aesara.tensor as tt
#import theano
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value='warn'

import matplotlib.pyplot as plt

class LIET:
    '''
    This class is intended for building LIET models using the pymc3 framework. 

    Suggested usage:

    x = LIET()
    x.load_annotation(**annot_dict)
    x.load_seq_data(**data_dict)
    x.set_priors(**priors_dict)
    x.build_model()
    
    with x.model:
        LIET_fit = pm.fit(<fit parameters>)
        post_samples = LIET_fit.sample(draws=10000)

    x.results['posteriors'] = post_samples

    x.posterior_summary()
    x.post_plot()
    x.model_plot()
    '''
    
    def __init__(self):

        # Initialize data variables
        annot_dict = {
            'gene_id': None,
            'chrom': None,
            'start': None,
            'stop': None,
            'strand': None
        }
        self.data = {
            'annot': annot_dict,
            'coord': np.array([], dtype='int64'),
            'pos_reads': np.array([], dtype='int64'),
            'neg_reads': np.array([], dtype='int64'),
            'shift': False,
            'pad': 0
        }

        # Initialize model variable for pymc3 Model() class
        self.model = None

        # Full priors map (including the offsets)
        self._p = {
            'mL': None, 
            'sL': None, 
            'tI': None, 
            'mT': None, 
            'sT': None, 
            'w': None,
            'mL_a': None,
            'sL_a': None,
            'tI_a': None
        }
        # Base priors map (pre offsets)
        self._o = {
            'mL': None, 
            'sL': None, 
            'tI': None, 
            'mT': None, 
            'sT': None, 
            'w': None,
            'mL_a': None,
            'sL_a': None,
            'tI_a': None
        }

        self.priors = {p:None for p in self._p.keys()}

        self.results = {
            'posteriors': None,
            'mL': None,
            'sL': None,
            'tI': None,
            'mT': None,
            'sT': None,
            'w': None,
            'mL_a': None,
            'sL_a': None,
            'tI_a': None
        }



    def load_annotation(
        self, 
        gene_id=None,
        chrom=None, 
        start=None, 
        stop=None, 
        strand=None
    ):
        strand_dict = {'+': 1, '-': -1, 1: 1, -1: -1}

        # Check that start is upstream of stop, regardless of strand
        if int(stop) < int(start):
            raise ValueError("Stop/start values have incorrect orientation.")
        
        annot_dict = {
            'gene_id': str(gene_id),
            'chrom': str(chrom),
            'start': int(start),
            'stop': int(stop),
            'strand': strand_dict[strand]
        }
        self.data['annot'] = annot_dict



    def load_seq_data(
        self, 
        coord=None,
#        positions=None, 
        pos_reads=None,
        neg_reads=None, 
        pad=0,
        shift=True,
    ):
        start = self.data['annot']['start']
        stop = self.data['annot']['stop']
        strand = self.data['annot']['strand']

        if shift is True:
            # Orient reads to positive strand and shift gene TSS to bp = 0
            if strand == 1:
                coord = np.array(coord) - start
                pos_reads = np.array(pos_reads) - start
                neg_reads = (np.array(neg_reads) - start) * (-1)               # This section still needs to be evaluated for correctness.
                neg_reads = np.flip(neg_reads, axis=0)
            elif strand == -1:
                coord = (np.array(coord) - stop) * (-1)
                coord = np.flip(coord, axis=0)
                pos_reads = np.array(pos_reads) - stop
                neg_reads = (np.array(neg_reads) - stop) * (-1)
                neg_reads = np.flip(neg_reads, axis=0)
            else:
                raise ValueError("Must specify +1 or -1 for strand.")

            self.data['shift'] = True

        self.data['coord'] = coord
        self.data['pos_reads'] = pos_reads
        self.data['neg_reads'] = neg_reads
        self.data['pad'] = pad



    def set_priors(self, **priors):
        '''
        <priors> input form: {<var_name_str>: <prior_input_format>, ...}

        Variable names: 
            Sense: 'mL', 'sL', 'tI', 'mT', 'sT', 'w'
            Anti-sense: 'mL_a', 'sL_a', 'tI_a'
            
            NOTE:
            If <prior_input_format> for 'sL_a' and 'tI_a' are set to 'None', 
            variables 'sL' and 'tI' will be used to model anti-sense strand 
            instead---e.g., loading uncertainty "sigma" is assumed to be the 
            same for both strands (recommended).

        Available priors and their input format:
            Constant:
                ['constant', [<value>]]
            Uniform:
                ['uniform', [<lower>, <upper>]]
            Exponential:
                ['exponential', [<lambda>, <offset>]]
            Normal:
                ['normal', [<mu>, <sigma>]]
            Gamma:
                ['gamma', [<mu>, <sigma>, <offset>]]
            Wald:
                ['wald', [<mu>, <lambda>, <alpha>]]
            Dirichlet:
                ['dirichlet', [<a_LI>, <a_E>, <a_T>, <a_B>]]
            None (see NOTE above):
                None

        Catagorical options for priors:
            Constant: 'mL', 'sL', 'tI', 'mT', 'sT'
            Positive-definite priors: 'sL', 'tI', 'sT'
            Real number priors: 'mL', 'mT'
            Dirichlet priors: 'w'

        offset/alpha: linear shift of respective dist. (exp, gamma, or wald)

        Uniform
            Support: [lower, upper]
            Mean: (lower + upper) / 2
            Var: (upper - lower)^2 / 12

        Exponential
            Support: [offset, +inf)
            OLD VERION: Mean: (1 / lambda) + offset
            OLD VERSION: Var: (1 / lambda)^2
            Mean : tau + offset
            Var: tau^2

        Normal
            Support: (-inf, +inf) 
            Mean: mu
            Var: sigma^2

        Gamma
            Support: (offset, +inf)
            Mean: mu + offset
            Var: sigma^2

        Wald
            Support: (alpha, +inf)
            Mean: mu + alpha
            OLD VERSION: Var: mu^3 / lambda
            Var: mu^3 * tau
        '''

        if priors:
            for p, prior in priors.items():
                if p not in self._p.keys():
                    raise ValueError(f"'{p}' not acceptable parameter name")
                else:
                    self.priors[p] = prior
        else:
            print(self.set_priors.__doc__)



    def build_model(self, antisense=True, background=True):
        '''Build the priors and model variables'''
        
        # Check that all the right variables are populated
        null_prior = [None for e in self.priors.values()]
        if self.data['coord'].size == 0 or self.priors.values() == null_prior:
            print('You must load data and priors before building model.')
            return 1

        # Initialize pymc3 model
        self.model = pm.Model()

        # Define priors for model parameters
        for var_name, prior in self.priors.items():
            
            if prior == None:
                prior_type = None
            else:
                prior_type = prior['dist']

            # Uniform prior
            if prior_type == 'uniform':
                with self.model:
                    start = prior['lower']
                    stop = prior['upper']
                    self._p[var_name] = pm.Uniform(
                        var_name, lower=start, upper=stop
                    )

            # Guassian prior
            elif prior_type == 'normal':
                with self.model:
                    norm_mu = prior['mu']
                    norm_sig = prior['sigma']
                    self._p[var_name] = pm.Normal(
                        var_name, mu=norm_mu, sigma=norm_sig
                    )

            # Exponential prior
            elif prior_type == 'exponential':
                exp_lam = 1 / prior['tau']
                offset = prior['offset']
                if offset == 0:
                    with self.model:
                        self._p[var_name] = pm.Exponential(
                            var_name, lam=exp_lam
                        )
                else:
                    with self.model:
                        self._o[var_name] = pm.Exponential(
                            var_name+'0', lam=exp_lam
                        )
                        self._p[var_name] = pm.Deterministic(
                            var_name, self._o[var_name] + offset
                        )

            # Gamma prior
            elif prior_type == 'gamma':
                gamma_mu = prior['mu']
                gamma_sig = prior['sigma']
                offset = prior['offset']
                if offset == 0:
                    with self.model:
                        self._p[var_name] = pm.Gamma(
                            var_name, mu=gamma_mu, sigma=gamma_sig
                        )
                else:
                    with self.model:
                        self._o[var_name] = pm.Gamma(
                            var_name+'0', mu=gamma_mu, sigma=gamma_sig
                        )
                        self._p[var_name] = pm.Deterministic(
                            var_name, self._o[var_name] + offset
                        )

            # Wald prior
            elif prior_type == 'wald':
                wald_mu = prior['mu']
                wald_lam = 1 / prior['tau']
                wald_alph = prior['alpha']
                with self.model:
                    self._p[var_name] = pm.Wald(
                        var_name, mu=wald_mu, lam=wald_lam, alpha=wald_alph
                    )

            # Dirichlet prior
            elif prior_type == 'dirichlet':
                alpha = [prior['alpha_LI'], prior['alpha_E'], 
                    prior['alpha_T'], prior['alpha_B']]

                with self.model:
                    self._p[var_name] = pm.Dirichlet(
                        var_name, a=np.array(alpha)
                    )
            
            # Constant 
            elif prior_type == 'constant':
                with self.model:
                    const = prior['value']
                    self._p[var_name] = pm.Deterministic(
                        var_name, tt.constant(const)
                    )

            # Check for None type prior
            # (used if anti-sense not being fit, sL==sL_a, and/or tI==tI_a )
            elif prior_type == None:
                if (antisense == True and 
                    var_name not in ['mL_a', 'sL_a', 'tI_a']):
                    print(prior_type)
                    raise ValueError(
                        (f"'{var_name}' must be one of the following: "
                        "'uniform', 'constant', 'normal', 'exponential', "
                        "'gamma', 'wald', or 'dirichlet'. 'None' type can be "
                        "specified for either prior 'sL_a' or 'tI_a', in "
                        "which case prior for 'sL' or 'tI' are used, "
                        "respectively."))
                else:
                    continue

            else:
                raise ValueError(
                    (f"'{var_name}' must be one of the following: "
                     "'uniform', 'constant', 'normal', 'exponential', 'gamma',"
                     " 'wald', or 'dirichlet'. 'None' type can be specified "
                     "for either prior 'sL_a' or 'tI_a', in which case prior "
                     "for 'sL' or 'tI' are used, respectively."))


        # Define model components (LI, E, T) --- sense strand
        with self.model:
            # Custom Elongation distribution ==================================
# OLDDEST APPROACH
#            def _emg_cdf(x, mu, sigma, tau):
#                # z = (x - mu) / sigma
#                def _norm_cdf(z):
#                    return 0.5 * (1 + tt.erf(z / tt.sqrt(2.0)))

#                z = (x - mu) / sigma
#                k = sigma / tau
#                exparg = 0.5*(k**2) - z*k
#                cdf =  _norm_cdf(z) - tt.exp(exparg) * _norm_cdf(z - k)

#                z = (x - mu) / sigma
#                invK = sigma / tau

#                exparg = invK * (0.5 * invK - z)
                # Sum of logs instead of product avoids overflow error
#                logprod = exparg + tt.log(_norm_cdf(z - invK))
                # Abs to avoid neg vals in diff at small prob (rounding error)
#                cdf = tt.abs_(_norm_cdf(z) - tt.exp(logprod))

#                return cdf

            # CDF/logCDF components
            def _emg_cdf(x, mu, sigma, tau):
                rv = pm.ExGaussian.dist(mu=mu,sigma=sigma, nu=tau)
                lcdf = pm.logcdf(rv, x)
                return tt.exp(lcdf)

            def _log_emg_cdf(x, mu, sigma, tau):
                rv = pm.ExGaussian.dist(mu=mu,sigma=sigma, nu=tau)
                lcdf = pm.logcdf(rv, x)
                return lcdf

            def _norm_sf(x, mu, sigma):
                arg = (x - mu) / (sigma * tt.sqrt(2.0))
                return 0.5 * tt.erfc(arg)

            def _log_norm_sf(x, mu, sigma):
                return pm.distributions.dist_math.normal_lccdf(mu, sigma, x)


            def elong_logp(x, mL, sL, tI, mT, sT):
                # Compute norm factor by integrating over entire distribution
                _n = 5 #number of stdevs for numerical normalization
                _min = tt.floor(tt.min([mL-_n*sL, mT-_n*sT]))
                _max = tt.ceil(tt.max([mL+_n*np.sqrt(sL**2+tI**2), mT+_n*sT]))

#                _min = tt.floor(tt.min([
#                    self._p['mL'] - _n*self._p['sL'], 
#                    self._p['mT'] - _n*self._p['sT']
#                ]))
#                _max = tt.ceil(tt.max([
#                    self._p['mL'] + _n*np.sqrt(self._p['sL']**2 
#                    + self._p['tI']**2), 
#                    self._p['mT'] + _n*self._p['sT']
#                ]))
                _x = tt.arange(_min, _max, dtype="int32")

                _norm_array = (
                    _emg_cdf(_x, mu=mL, sigma=sL, tau=tI) 
                    *_norm_sf(_x, mu=mT, sigma=sT)
                )
#                _norm_array = (
#                    _emg_cdf(
#                        _x, 
#                        mu=self._p['mL'], 
#                        sigma=self._p['sL'], 
#                        tau=self._p['tI']
#                    )
#                    * _norm_sf(
#                        _x, 
#                        mu=self._p['mT'], 
#                        sigma=self._p['sT']
#                    )
#                )
                _log_norm_factor = tt.log(tt.sum(_norm_array))

                # Unnormalized dist values (log(CDF*SF) = log(CDF) + log(SF))
                _log_unscaled = (
                    _log_emg_cdf(x, mu=mL, sigma=sL, tau=tI)
                    +_log_norm_sf(x, mu=mT, sigma=sT)
                )
#                _log_unscaled = (
#                    _log_emg_cdf(
#                        x, 
#                        mu=self._p['mL'], 
#                        sigma=self._p['sL'], 
#                        tau=self._p['tI']
#                    ) 
#                    + _log_norm_sf(
#                        x, 
#                        mu=self._p['mT'],
#                        sigma=self._p['sT']
#                    )
#                )

                # Normalize distribution in logscale
                log_pdf = _log_unscaled - _log_norm_factor
                                                                            # NOT SURE IF I NEED THIS BOUNDING
#                log_pdf = pm.distributions.dist_math.bound(_log_unscaled - _log_norm_factor, self._p['mL'] < self._p['mT'])

                return log_pdf
                #==============================================================
            # Debugging print statement (can remove later)
#            mL_print = tt.printing.Print('mL')(self._p['mL'])

            # Distribution for the Loading/Initiation phase (native to pymc)
            LI_pdf = pm.ExGaussian.dist(
#                mu=mL_print,
                mu=self._p['mL'], 
                sigma=self._p['sL'], 
                nu=self._p['tI']
            )

            # Convert Aesara log-prob func into pymc distribution variable
            E_pdf = pm.DensityDist.dist(
                self._p['mL'],
                self._p['sL'],
                self._p['tI'],
                self._p['mT'],
                self._p['sT'],
                logp=elong_logp,
                class_name='E_pdf'
            )
#            E_pdf = pm.DensityDist.dist(class_name='E_pdf', logp=elong_logp)
            
            # Distribution for the Termination phase (native to pymc)
            T_pdf = pm.Normal.dist(mu=self._p['mT'], sigma=self._p['sT'])

        # Strand data dict used to reference self.data for 'observed' kwargs
        strand_ref = {1: 'pos_reads', -1: 'neg_reads'}
        sense_reads = strand_ref[self.data['annot']['strand']]
        antisense_reads = strand_ref[-1*self.data['annot']['strand']]

        # Define sense-strand full model (with or without background)
        if background == True and self.priors['w']['alpha_B'] != 0:
            
            xmin = -1 + min(
                self.data['pos_reads'].min(), 
                self.data['neg_reads'].min()
            )
            xmax = 1 + max(
                self.data['pos_reads'].max(), 
                self.data['neg_reads'].max()
            )
            with self.model:
                back_pdf = pm.Uniform.dist(lower=xmin, upper=xmax)

            components = [LI_pdf, E_pdf, T_pdf, back_pdf]
        else:
            components = [LI_pdf, E_pdf, T_pdf]

        with self.model:
            LIET_pdf = pm.Mixture(
                    'LIET_pdf',
                    w=self._p['w'],
                    comp_dists=components,
                    observed=self.data[sense_reads]
                )

        # Define antisense-strand model (w/ or w/o bckgrnd or sep sL/tI priors)
        if antisense == True:
            if self.priors['mL_a'] != None:
                m_a = self._p['mL_a']
            else:
                m_a = self._p['mL']

            if self.priors['sL_a'] != None:
                s_a = self._p['sL_a']
            else:
                s_a = self._p['sL']

            if self.priors['tI_a'] != None:
                t_a = self._p['tI_a']
            else:
                t_a = self._p['tI']
            
            if background == True and self.priors['w']['alpha_B'] != 0:
                with self.model:
                    LI_a_pdf = pm.ExGaussian.dist(
                        mu=m_a, 
                        sigma=s_a, 
                        nu=t_a
                    )
                    components = [LI_a_pdf, back_pdf]

                w_a = [self.priors['w']['alpha_LI'], 
                    self.priors['w']['alpha_B']]
                
                with self.model:
                    w_a = pm.Dirichlet('w_a', a=np.array(w_a))

                    LIET_a_pdf = pm.Mixture(
                        'LIET_a_pdf', 
                        w=w_a, 
                        comp_dists=components, 
                        observed=self.data[antisense_reads]
                    )

            else:
                with self.model:
                    LIET_a_pdf = pm.ExGaussian(
                        'LIET_a_pdf',
                        mu=self.m_a,
                        sigma=s_a,
                        nu=t_a,
                        observed=self.data[antisense_reads]
                    )



    def posterior_summary(self):
        '''
        This function requires that you've already run a pymc3 fit routine and 
        `self.results['posteriors']` has been assigned a fit "trace" 
        (in the parlance of pymc3). The trace format is a array of posterior 
        samples from the model fit. Each array element is a `dict` containing 
        a sample from the posteriors of each parameter. This function writes 
        summary statistics of each posterior to `self.results['opt_params']`. 
        '''

        def kde_mode(samples, tol=1e-2):
            '''
            Computes the mode of the distribution after applying a guassian kde
            '''
            smin = min(samples)
            smax = max(samples)
            N = int((smax - smin)/tol)
            
            samp_kde = sp.stats.gaussian_kde(samples)
            x = np.linspace(smin, smax, N)
            y = samp_kde.pdf(x)

            mode = max(zip(x,y), key = lambda x : x[1])[0]

            return mode


        for p in self._p.keys():

            try:
                # Pad the w_b weight with zeros, if it was not part of fit
                if p == 'w':
                    samps = self.results['posteriors'][p][:]
                    zeros = np.zeros((len(samps), 1))
                    samps = np.concatenate((samps, zeros), axis=1)

                else:
                    samps = self.results['posteriors'][p][:]

                # Compute the stats for parameter p
                mean = np.mean(samps, axis=0)
                median = np.median(samps, axis=0)
                std = sp.stats.tstd(samps, axis=0)
#                mode = sp.stats.mode(samps, axis=0)[0][0]
                mode = kde_mode(samps)
                skew = sp.stats.skew(samps, axis=0)
                skewtest = sp.stats.skewtest(samps, axis=0).pvalue

                self.results[p] = {
                    'mean': mean,
                    'median': median,
                    'std': std,
                    'mode': mode,
                    'skew': skew,
                    'skewtest': skewtest
                }

            except:
                print(f"WARNING: Posterior for {p} not present.")
                continue

    
    
    def prior_plot(self):
        '''
        Plots the prior distributions. Requires that the model has been built
        '''

        if self.model:
            
            # Select non-None type parameters
            non_none_params = [p for p, v in self._p.items() if v != None]

            # Sample priors
            with self.model:
                prior_samp = pm.sample_prior_predictive(
                    samples=10000, 
                    var_names=non_none_params
                    #var_names = self._p.keys()
                )

            # Plot priors
            plt.rcParams['figure.figsize'] = 11, 15
            ax = []
            for i, p in enumerate(['mL', 'sL', 'tI', 'mT', 'sT']):

                subplot_int = int("".join(['61', str(i+1)]))
                ax.append(plt.subplot(subplot_int))
                ax[i].hist(
                    prior_samp[p],
                    bins='doane',
                    alpha=0.6, 
                    density=True
                )
                ax[i].legend([" ".join([p, 'prior'])])
        
        else:
            print("WARNING: Must build model prior to sampling priors.")



    def post_plot(self, save=None):
        '''
        Plots the posterior distributions and (if computed) identifies the 
        statistics for each parameter.

        TODO: Fix the save kwarg, add if statement to check existence of stats for axvlines
        '''

        if self.results['posteriors']:
            
            plt.rcParams['figure.figsize'] = 11, 15

            # Plot mu_L
            ax1 = plt.subplot(611)
            post = self.results['posteriors']['mL'][:]
            ax1.hist(post, bins='auto', alpha=0.6, density=True)
            ax1.legend(['mL posterior'])
            mean = self.results['mL']['mean']
            std = self.results['mL']['std']
            ax1.axvline(x=mean, color='k')
            ax1.axvline(x=mean-std, color='k', alpha=0.5)
            ax1.axvline(x=mean+std, color='k', alpha=0.5)

            # Plot sigma_L
            ax2 = plt.subplot(612)
            post = self.results['posteriors']['sL'][:]
            ax2.hist(post, bins='auto', alpha=0.6, density=True)
            ax2.legend(['sL posterior'])
            mean = self.results['sL']['mean']
            std = self.results['sL']['std']
            ax2.axvline(x=mean, color='k')
            ax2.axvline(x=mean-std, color='k', alpha=0.5)
            ax2.axvline(x=mean+std, color='k', alpha=0.5)

            # Plot tau_I
            ax3 = plt.subplot(613)
            post = self.results['posteriors']['tI'][:]
            ax3.hist(post, bins='auto', alpha=0.6, density=True)
            ax3.legend(['tI posterior'])
            mean = self.results['tI']['mean']
            std = self.results['tI']['std']
            ax3.axvline(x=mean, color='k')
            ax3.axvline(x=mean-std, color='k', alpha=0.5)
            ax3.axvline(x=mean+std, color='k', alpha=0.5)

            # Plot mu_T
            ax4 = plt.subplot(614)
            post = self.results['posteriors']['mT'][:]
            ax4.hist(post, bins='auto', alpha=0.6, density=True)
            ax4.legend(['mT posterior'])
            mean = self.results['mT']['mean']
            std = self.results['mT']['std']
            ax4.axvline(x=mean, color='k')
            ax4.axvline(x=mean-std, color='k', alpha=0.5)
            ax4.axvline(x=mean+std, color='k', alpha=0.5)
            
            # Plot sigma_T
            ax5 = plt.subplot(615)
            post = self.results['posteriors']['sT'][:]
            ax5.hist(post, bins='auto', alpha=0.6, density=True)
            ax5.legend(['sT posterior'])
            mean = self.results['sT']['mean']
            std = self.results['sT']['std']
            ax5.axvline(x=mean, color='k')
            ax5.axvline(x=mean-std, color='k', alpha=0.5)
            ax5.axvline(x=mean+std, color='k', alpha=0.5)

            # Plot weights
            ax6 = plt.subplot(616)
            if self.priors['w'][1][3] == 0:
                wLI, wE, wT = zip(*self.results['posteriors']['w'])
            else:
                wLI, wE, wT, wB = zip(*self.results['posteriors']['w'])

            ax6.hist(wLI, bins='auto', color='C0', alpha=0.6, density=True)
            ax6.hist(wE, bins='auto', color='C1', alpha=0.6, density=True)
            ax6.hist(wT, bins='auto', color='C2', alpha=0.6, density=True)
            ax6.legend(['wLI', 'wE', 'wT'])
            
            if self.priors['w'][-1] == 0:
                mLI, mE, mT = self.results['w']['mean']
            else:
                mLI, mE, mT, mB = self.results['w']['mean']
                ax6.axvline(x=mB, color='k')

            ax6.axvline(x=mLI, color='C0')
            ax6.axvline(x=mE, color='C1')
            ax6.axvline(x=mT, color='C2')

        else:
            print(
                "WARNING: No posterior samples exist in `self.results`. Must "
                "first run fitting routine and assign posterior samples to "
                "`self.results['posteriors']."
            )



    def model_plot(
            self, 
            stat='mean', 
            bins='auto', 
            data=True, 
            sense=True,
            antisense=False,
            save=None):
        '''
        Plot the model pdf that results from fitting. `stat` specifies which 
        value should be used for the parameter value. Option is either 'mean' 
        or 'mode'.
        '''
        # Check that parameter stats exist.
        if self.results['mL']:

            plt.rcParams['figure.figsize'] = 10, 7

            mL = self.results['mL'][stat]
            sL = self.results['sL'][stat]
            tI = self.results['tI'][stat]
            mT = self.results['mT'][stat]
            sT = self.results['sT'][stat]
            w = self.results['w'][stat]
            print(w)
            print(len(w))
            if len(w) == 3:
                print(w)
                print(len(w))
                w.extend([0])
                print(w)

            pdf = ds.gene_model(
                self.data['coord'],
                mu0 = mL,
                sig0 = sL,
                tau0 = tI,
                mu1 = mT,
                sig1 = sT,
                weights = w,
                rvs = False
            )

            plt.plot(self.data['coord'], pdf)
            
            if data:
                if self.data['annot']['strand'] == 1:
                    plt.hist(self.data['pos_reads'], density=True, bins=bins)
                    plt.legend(['LIET fit', 'Data'])
                else:
                    plt.hist(self.data['neg_reads'], density=True, bins=bins)
                    plt.legend(['LIET fit', 'Data'])

        else:
            print(
                "WARNING: Optimal parameter values not in `self.results`. "
                "Must first run fitting routine and posterior_summary()."
            )

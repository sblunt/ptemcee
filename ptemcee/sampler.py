# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ['make_ladder', 'Sampler']

import attr
import itertools
import numpy as np

from numpy.random.mtrand import RandomState

from . import util, chain, ensemble


def make_ladder(ndim, ntemps=None, Tmax=None):
    """
    Returns a ladder of :math:`\\beta \\equiv 1/T` under a geometric spacing that is determined by the
    arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:

    Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
    this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
    <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
    ``ntemps`` is also specified.

    :param ndim:
        The number of dimensions in the parameter space.

    :param ntemps: (optional)
        If set, the number of temperatures to generate.

    :param Tmax: (optional)
        If set, the maximum temperature for the ladder.

    Temperatures are chosen according to the following algorithm:

    * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
      information).
    * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
      posterior would have a 25% temperature swap acceptance ratio.
    * If ``Tmax`` is specified but not ``ntemps``:

      * If ``Tmax = inf``, raise an exception (insufficient information).
      * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.

    * If ``Tmax`` and ``ntemps`` are specified:

      * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
      * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.

    """

    if type(ndim) != int or ndim < 1:
        raise ValueError('Invalid number of dimensions specified.')
    if ntemps is None and Tmax is None:
        raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
    if Tmax is not None and Tmax <= 1:
        raise ValueError('``Tmax`` must be greater than 1.')
    if ntemps is not None and (type(ntemps) != int or ntemps < 1):
        raise ValueError('Invalid number of temperatures specified.')

    tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                      2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                      2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                      1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                      1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                      1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                      1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                      1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                      1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                      1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                      1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                      1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                      1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                      1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                      1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                      1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                      1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                      1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                      1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                      1.26579, 1.26424, 1.26271, 1.26121,
                      1.25973])

    if ndim > tstep.shape[0]:
        # An approximation to the temperature step at large
        # dimension
        tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
    else:
        tstep = tstep[ndim-1]

    appendInf = False
    if Tmax == np.inf:
        appendInf = True
        Tmax = None
        ntemps = ntemps - 1

    if ntemps is not None:
        if Tmax is None:
            # Determine Tmax from ntemps.
            Tmax = tstep ** (ntemps - 1)
    else:
        if Tmax is None:
            raise ValueError('Must specify at least one of ``ntemps'' and '
                             'finite ``Tmax``.')

        # Determine ntemps from Tmax.
        ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

    betas = np.logspace(0, -np.log10(Tmax), ntemps)
    if appendInf:
        # Use a geometric spacing, but replace the top-most temperature with
        # infinity.
        betas = np.concatenate((betas, [0]))

    return betas


@attr.s(slots=True, frozen=True)
class LikePriorEvaluator(object):
    """
    Wrapper class for logl and logp.

    """

    logl = attr.ib()
    logp = attr.ib()
    logl_args = attr.ib(factory=list)
    logp_args = attr.ib(factory=list)
    logl_kwargs = attr.ib(factory=dict)
    logp_kwargs = attr.ib(factory=dict)

    def __call__(self, x):
        lp = self.logp(x, *self.logp_args, **self.logp_kwargs)
        if np.isnan(lp):
            raise ValueError('Prior function returned NaN.')

        if lp == float('-inf'):
            # Can't return -inf, since this messes with beta=0 behaviour.
            ll = 0
        else:
            ll = self.logl(x, *self.logl_args, **self.logl_kwargs)
            if np.isnan(ll).any():
                raise ValueError('Log likelihood function returned NaN.')

        return ll, lp


@attr.s(slots=True, frozen=True)
class Sampler(object):
    # Mandatory parameters.
    nwalkers = attr.ib(converter=int)
    ndim = attr.ib(converter=int)

    logl = attr.ib()
    logp = attr.ib()
    logl_args = attr.ib(converter=list, factory=list)
    logp_args = attr.ib(converter=list, factory=list)
    logl_kwargs = attr.ib(converter=dict, factory=dict)
    logp_kwargs = attr.ib(converter=dict, factory=dict)

    betas = attr.ib(default=None)

    # Tuning parameters.
    adaptive = attr.ib(converter=bool, default=False)
    adaptation_lag = attr.ib(converter=int, default=10000)
    adaptation_time = attr.ib(converter=int, default=100)
    scale_factor = attr.ib(converter=float, default=2)

    _mapper = attr.ib(default=map)
    _evaluator = attr.ib(type=LikePriorEvaluator, init=False, default=None)
    _data = attr.ib(type=np.ndarray, init=False, default=None)

    @nwalkers.validator
    def _validate_nwalkers(self, attribute, value):
        if value % 2 != 0:
            raise ValueError('The number of walkers must be even.')
        if self.nwalkers < 2 * self.dim:
            raise ValueError('The number of walkers must be greater than ``2*dimension``.')

        self.pool = pool
        if threads > 1 and pool is None:
            self.pool = multi.Pool(threads)

        self.reset()

    def reset(self, random=None, betas=None, time=None):
        """
        Clear the ``time``, ``chain``, ``logposterior``,
        ``loglikelihood``,  ``acceptance_fraction``,
        ``tswap_acceptance_fraction`` stored properties.

        """

        # Reset chain.
        self._chain = None
        self._logposterior = None
        self._loglikelihood = None
        self._beta_history = None

        # Reset sampler state.
        self._time = 0
        self._p0 = None
        self._logposterior0 = None
        self._loglikelihood0 = None
        if betas is not None:
            self._betas = betas

        self.nswap = np.zeros(self.ntemps, dtype=float)
        self.nswap_accepted = np.zeros(self.ntemps, dtype=float)

        self.nprop = np.zeros((self.ntemps, self.nwalkers), dtype=float)
        self.nprop_accepted = np.zeros((self.ntemps, self.nwalkers), dtype=float)

        if random is not None:
            self._random = random
        if time is not None:
            self._time = time

    def run_mcmc(self, *args, **kwargs):
        """
        Identical to ``sample``, but returns the final ensemble and discards intermediate ensembles.

        """
        for x in self.sample(*args, **kwargs):
            pass
        return x

    def sample(self, p0=None,
               iterations=1, thin=1,
               storechain=True, adapt=False,
               swap_ratios=False):
        """
        Advance the chains ``iterations`` steps as a generator.

        :param p0:
            The initial positions of the walkers.  Shape should be
            ``(ntemps, nwalkers, dim)``.  Can be omitted if resuming
            from a previous run.

        :param iterations: (optional)
            The number of iterations to perform.

        :param thin: (optional)
            The number of iterations to perform between saving the
            state to the internal chain.

        :param storechain: (optional)
            If ``True`` store the iterations in the ``chain``
            property.

        :param adapt: (optional)
            If ``True``, the temperature ladder is dynamically adapted as the sampler runs to
            achieve uniform swap acceptance ratios between adjacent chains.  See `arXiv:1501.05823
            <http://arxiv.org/abs/1501.05823>`_ for details.

        :param swap_ratios: (optional)
            If ``True``, also yield the instantaneous inter-chain acceptance ratios in the fourth
            element of the yielded tuple.

        At each iteration, this generator yields

        * ``p``, the current position of the walkers.

        * ``logpost``, the current posterior values for the walkers.

        * ``loglike``, the current likelihood values for the walkers.

        * ``ratios``, the instantaneous inter-chain acceptance ratios (if requested).

        """

        # Set initial walker positions.
        if p0 is not None:
            # Start anew.
            self._p0 = p = np.array(p0).copy()
            self._logposterior0 = None
            self._loglikelihood0 = None
        elif self._p0 is not None:
            # Now, where were we?
            p = self._p0
        else:
            raise ValueError('Initial walker positions not specified.')

        # Check for dodgy inputs.
        if np.any(np.isinf(p)):
            raise ValueError('At least one parameter value was infinite.')
        if np.any(np.isnan(p)):
            raise ValueError('At least one parameter value was NaN.')

        # If we have no likelihood or prior values, compute them.
        if self._logposterior0 is None or self._loglikelihood0 is None:
            logl, logp = self._evaluate(p)
            logpost = self._tempered_likelihood(logl) + logp

            self._loglikelihood0 = logl
            self._logposterior0 = logpost
        else:
            logl = self._loglikelihood0
            logpost = self._logposterior0

        if (logpost == -np.inf).any():
            raise ValueError('Attempting to start with samples outside posterior support.')

        # Expand the chain in advance of the iterations
        if storechain:
            isave = self._expand_chain(iterations // thin)

        for i in range(iterations):
            for j in [0, 1]:
                # Get positions of walkers to be updated and walker to be sampled.
                jupdate = j
                jsample = (j + 1) % 2
                pupdate = p[:, jupdate::2, :]
                psample = p[:, jsample::2, :]

                zs = np.exp(self._random.uniform(low=-np.log(self.a),
                                                 high=np.log(self.a),
                                                 size=(self.ntemps, self.nwalkers//2)))

                qs = np.zeros((self.ntemps, self.nwalkers//2, self.dim))
                for k in range(self.ntemps):
                    js = self._random.randint(0, high=self.nwalkers // 2,
                                              size=self.nwalkers // 2)
                    qs[k, :, :] = psample[k, js, :] + zs[k, :].reshape(
                        (self.nwalkers // 2, 1)) * (pupdate[k, :, :] -
                                                   psample[k, js, :])

                qslogl, qslogp = self._evaluate(qs)
                qslogpost = self._tempered_likelihood(qslogl) + qslogp

                logpaccept = self.dim*np.log(zs) + qslogpost \
                    - logpost[:, jupdate::2]
                logr = np.log(self._random.uniform(low=0.0, high=1.0,
                                                   size=(self.ntemps,
                                                         self.nwalkers//2)))

                accepts = logr < logpaccept
                accepts = accepts.flatten()

                pupdate.reshape((-1, self.dim))[accepts, :] = \
                    qs.reshape((-1, self.dim))[accepts, :]
                logpost[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslogpost.reshape((-1,))[accepts]
                logl[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslogl.reshape((-1,))[accepts]

                accepts = accepts.reshape((self.ntemps, self.nwalkers//2))

                self.nprop[:, jupdate::2] += 1.0
                self.nprop_accepted[:, jupdate::2] += accepts

            p, ratios = self._temperature_swaps(self._betas, p, logpost, logl)

            # TODO Should the notion of a "complete" iteration really include the temperature
            # adjustment?
            if adapt and self.ntemps > 1:
                dbetas = self._get_ladder_adjustment(self._time, self._betas, ratios)
                self._betas += dbetas
                logpost += self._tempered_likelihood(logl, betas=dbetas)

            if (self._time + 1) % thin == 0:
                if storechain:
                    self._chain[:, :, isave, :] = p
                    self._logposterior[:, :, isave] = logpost
                    self._loglikelihood[:, :, isave] = logl
                    self._beta_history[:, isave] = self._betas
                    isave += 1

            self._time += 1
            if swap_ratios:
                yield p, logpost, logl, ratios
            else:
                yield p, logpost, logl

    def _evaluate(self, ps):
        mapf = map if self.pool is None else self.pool.map
        results = list(mapf(self._likeprior, ps.reshape((-1, self.dim))))

        logl = np.fromiter((r[0] for r in results), float,
                           count=len(results)).reshape((self.ntemps, -1))
        logp = np.fromiter((r[1] for r in results), float,
                           count=len(results)).reshape((self.ntemps, -1))

        return logl, logp

    def _tempered_likelihood(self, logl, betas=None):
        """
        Compute tempered log likelihood.  This is usually a mundane multiplication, except for the
        special case where beta == 0 *and* we're outside the likelihood support.

        Here, we find a singularity that demands more careful attention; we allow the likelihood to
        dominate the temperature, since wandering outside the likelihood support causes a discontinuity.

        """

        if betas is None:
            betas = self._betas
        betas = betas.reshape((-1, 1))

        with np.errstate(invalid='ignore'):
            loglT = logl * betas
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def _temperature_swaps(self, betas, p, logpost, logl):
        """
        Perform parallel-tempering temperature swaps on the state
        in ``p`` with associated ``logpost`` and ``logl``.

        """
        ntemps = len(betas)
        ratios = np.zeros(ntemps - 1)
        for i in range(ntemps - 1, 0, -1):
            bi = betas[i]
            bi1 = betas[i - 1]

            dbeta = bi1 - bi

            iperm = self._random.permutation(self.nwalkers)
            i1perm = self._random.permutation(self.nwalkers)

            raccept = np.log(self._random.uniform(size=self.nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[i - 1, i1perm])

            self.nswap[i] += self.nwalkers
            self.nswap[i - 1] += self.nwalkers

            asel = (paccept > raccept)
            nacc = np.sum(asel)

            self.nswap_accepted[i] += nacc
            self.nswap_accepted[i - 1] += nacc

            ratios[i - 1] = nacc / self.nwalkers

            ptemp = np.copy(p[i, iperm[asel], :])
            logltemp = np.copy(logl[i, iperm[asel]])
            logprtemp = np.copy(logpost[i, iperm[asel]])

            p[i, iperm[asel], :] = p[i - 1, i1perm[asel], :]
            logl[i, iperm[asel]] = logl[i - 1, i1perm[asel]]
            logpost[i, iperm[asel]] = logpost[i - 1, i1perm[asel]] \
                - dbeta * logl[i - 1, i1perm[asel]]

            p[i - 1, i1perm[asel], :] = ptemp
            logl[i - 1, i1perm[asel]] = logltemp
            logpost[i - 1, i1perm[asel]] = logprtemp + dbeta * logltemp

        return p, ratios

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.

        """

        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.adaptation_lag / (time + self.adaptation_lag)
        kappa = decay / self.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0

    def _expand_chain(self, nsave):
        """
        Expand ``self._chain``, ``self._logposterior``,
        ``self._loglikelihood``, and ``self._beta_history``
        ahead of run to make room for new samples.

        :param nsave:
            The number of additional iterations for which to make room.

        :return ``isave``:
            Returns the index at which to begin inserting new entries.

        """

        if self._chain is None:
            isave = 0
            self._chain = np.zeros((self.ntemps, self.nwalkers, nsave,
                                    self.dim))
            self._logposterior = np.zeros((self.ntemps, self.nwalkers, nsave))
            self._loglikelihood = np.zeros((self.ntemps, self.nwalkers,
                                            nsave))
            self._beta_history = np.zeros((self.ntemps, nsave))
        else:
            betas = util._ladder(self.betas)

        object.__setattr__(self, 'betas', betas)
        object.__setattr__(self, '_evaluator',
                           LikePriorEvaluator(logl=self.logl,
                                              logp=self.logp,
                                              logl_args=self.logl_args,
                                              logp_args=self.logp_args,
                                              logl_kwargs=self.logl_kwargs,
                                              logp_kwargs=self.logp_kwargs))

    def ensemble(self, x, random=None):
        if random is None:
            random = RandomState()
        elif not isinstance(random, RandomState):
            raise TypeError('Invalid random state.')

        config = ensemble.EnsembleConfiguration(adaptation_lag=self.adaptation_lag,
                                                adaptation_time=self.adaptation_time,
                                                scale_factor=self.scale_factor,
                                                evaluator=self._evaluator)
        return ensemble.Ensemble(x=x,
                                 betas=self.betas.copy(),
                                 config=config,
                                 adaptive=self.adaptive,
                                 random=random,
                                 mapper=self._mapper)

    def sample(self, x, random=None, thin_by=None):
        """
        Return a stateless iterator.

        """

        if thin_by is None:
            thin_by = 1

        # Don't yield the starting state.
        ensemble = self.ensemble(x, random)
        while True:
            for _ in range(thin_by):
                ensemble.step()
            yield ensemble

    def chain(self, x, random=None, thin_by=None):
        """
        Create a stateful chain that stores its history.

        """
        return chain.Chain(self.ensemble(x, random), thin_by)

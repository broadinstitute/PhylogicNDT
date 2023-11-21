import collections
import sys
import logging
import functools
import random
import numpy as np

from scipy import stats
from scipy import cluster as scipy_cluster
# import sklearn.manifold as manifold
from sklearn import preprocessing
from math import lgamma
from scipy.optimize import minimize
import itertools

# Logsumexp options
# always import as double safe method (slow)
try:
    from scipy.misc import logsumexp as logsumexp_scipy
except ImportError:
    from scipy.special import logsumexp as logsumexp_scipy


import pkgutil

if pkgutil.find_loader('sselogsumexp') is not None:
    logging.info("Using fast logsumexp")
    from sselogsumexp import logsumexp
else:
    logging.info("Using scipy (slower) logsumexp")
    try:
        from scipy.misc import logsumexp as logsumexp
    except ImportError:
        from scipy.special import logsumexp as logsumexp


class DpEngine:

    def __init__(self, data, N_iter, Pi_k, use_fixed=False, co_assign_flag=False, ignore_nan=False, tsne=True,
                 mode=None, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed=seed)
        self.data = data
        self.num_bins = data.n_bins
        self.n_samples = data.n_samples
        self.logprior = np.log(np.zeros((self.n_samples, self.num_bins), dtype=np.float32) + 1.0 / (self.num_bins))
        self.clusterlist = []
        self.mutations = []
        self.results = ClusteringResults()
        self.alpha = None
        self.eta = None

        if mode:
            logging.error("Mode explicitly in options. This functionality is disabled, using hclust for assignment.")
        if ignore_nan:
            logging.error("Ignore nan specified, but the input should be a CCF histogram class which controls for NaNs internally. "
                          "This functionality is redundant, and removed.")
        if co_assign_flag:
            logging.error("co assignment matrix not calculated, but now outputs are available to generate this if desired")

        if use_fixed:
            logging.error("fixed mutations ignored. this feature has been removed")

        DP_prior = init_dp_prior(len(data._hist_array), Pi_k)

        Pi_gamma_a = DP_prior["a"]
        Pi_gamma_b = DP_prior["b"]

        self.gamma_prior = [Pi_gamma_a, Pi_gamma_b]

        logging.info("Calculated gamma prior for alpha as a={},b={}".format(Pi_gamma_a, Pi_gamma_b))

        self.alpha = stats.gamma.rvs(Pi_gamma_a, scale=1. / Pi_gamma_b)

        if N_iter == 0:
            return None

        mut_iter = data.iteritems()
        s_cluster = DP_cluster(self, DP_item(next(mut_iter)[0], self))

        for label, mut_hist in mut_iter:
            # make cluster and mutation. they are automatically updated to the list.
            s_cluster += DP_item(label, self)

        # Create mutation and cluster objects

        N_burn = N_iter // 2
        logging.info("Starting {} Iterations".format(N_iter))

        for i in range(N_iter):  # run all
            if not i % 50:  # every 50-th iteration
                logging.info("Iter {} n_clusters(alpha)".format(i))
            self.one_iteration()

        logging.info("Completed Iterations")

        var_loc, K = summarize_mut_locations(self.results, N_burn)

        cluster_out = self.tree_cluster_DP(co_assign_flag, K, var_loc=var_loc, tsne=tsne,
                                           grid_size=self.num_bins)  # co-assignment matrix is global member of Mut class.
        cluster_out["var_loc"] = var_loc
        logging.debug("assignments:" + str(cluster_out["assign"]))
        self.results.debug_outputs.append(self.results.assign[:])
        for key, value in cluster_out.items():
            setattr(self.results, key, value)

    def get_results(self):
        return self.results, {}

    def one_iteration(self, resample=True):
        for mut in self.mutations:
            skip_count = 1
            loglik = np.ones((self.n_clusters + 1, self.n_samples), dtype=np.float64) * -np.inf
            const_array = np.zeros((self.n_clusters + 1,), dtype=np.float64)

            for cluster_idx, cluster in enumerate(self.clusterlist):
                ## if the current point is the only thing in the cluster...
                # This seems to work empirically (as well as theoretically)
                if len(cluster) == 1 and mut.assigned_to == cluster:
                    # skip_count+=1 #at most 2
                    continue
                stay_in_clust_const = len(cluster) / float(self.n_muts - 1 + self.alpha)

                const_array[cluster_idx] = np.log(stay_in_clust_const)

                if mut not in cluster:  # TODO: redefine in
                    loglik[cluster_idx] = self.logsum_of_marginals_per_sample(cluster.normed_hist + mut.loghist)
                else:
                    loglik[cluster_idx] = self.logsum_of_marginals_per_sample(
                        self.normalize_loghist_with_prior(cluster - mut) + mut.loghist)

            open_new_clust_const = self.alpha / float(self.n_muts - 1 + self.alpha)

            prior = np.clip(np.exp(self.logprior) - np.exp(
                functools.reduce(lambda x, y: np.maximum(x, y), [z.normed_hist for z in self.clusterlist])), a_min=1e-40,
                            a_max=1.)

            loglik[-1] = self.logsum_of_marginals_per_sample(
                mut.loghist + self.normalize_loghist_with_prior(np.log(prior)))
            const_array[-1] = np.log(open_new_clust_const)

            # c_loglik = np.sum(c_loglik, axis = 1)
            loglik = np.sum(loglik, axis=1)  # + const_array

            loglik = loglik - logsumexp_scipy(loglik)
            loglik = loglik + const_array

            c_lik = np.exp(loglik - logsumexp_scipy(loglik))

            new_cluster_idx = np.nonzero(np.random.multinomial(1, c_lik) == 1)[0][0]
            if new_cluster_idx == self.n_clusters:  # new cluster
                mut.assigned_to -= mut
                DP_cluster(self, mut)  # create new cluster, lists updated automatically
            else:
                new_cluster = self.clusterlist[new_cluster_idx]
                mut.assigned_to -= mut
                new_cluster += mut

        cluster_counter = itertools.count()
        next(cluster_counter)
        real_index = dict([[x.id, next(cluster_counter)] for x in self.clusterlist])
        self.results.assign.append([real_index[x.assigned_to.id] for x in self.mutations])
        self.results.alpha.append(self.alpha)
        self.results.eta.append(self.eta)
        self.results.cluster_loghistograms.append([cluster.normed_hist for cluster in self.clusterlist])
        self.results.cluster_positions.append(
            [[np.argmax(x) for x in cluster.normed_hist] for cluster in self.clusterlist])
        self.results.clust_prop.append([len(cluster) / float(self.n_muts) for cluster in self.clusterlist])
        self.results.clust_size.append([len(cluster) for cluster in self.clusterlist])
        self.results.K.append(self.n_clusters)

        print("{}({});".format(self.n_clusters, round(self.alpha, 1)),)
        sys.stdout.flush()

        if resample:
            ##resample alpha
            self.eta = stats.beta.rvs(self.alpha + 1, self.n_muts)
            self.alpha = sample_gamma_cond_N_k(self.n_muts, self.n_clusters, self.eta,
                                               self.gamma_prior)  ## Escobar and West 1995

    def one_iteration_fix_k(self):
        random.shuffle(self.mutations)
        for mut in self.mutations:
            skip_count = 1
            loglik = np.ones((self.n_clusters, self.n_samples), dtype=np.float64) * -np.inf
            const_array = np.zeros((self.n_clusters,), dtype=np.float64)

            if len(mut.assigned_to) == 1: continue  # don't reassign last mutation

            for cluster_idx, cluster in enumerate(self.clusterlist):
                ## if the current point is the only thing in the cluster...
                # This seems to work empirically (as well as theoretically)
                if len(cluster) == 1 and mut.assigned_to == cluster:
                    continue
                stay_in_clust_const = len(cluster) / float(self.n_muts - 1 + self.alpha)

                const_array[cluster_idx] = np.log(stay_in_clust_const)

                if mut not in cluster:  # TODO: redefine in
                    loglik[cluster_idx] = self.logsum_of_marginals_per_sample(cluster.normed_hist + mut.loghist)
                else:
                    loglik[cluster_idx] = self.logsum_of_marginals_per_sample(
                        self.normalize_loghist_with_prior(cluster - mut) + mut.loghist)

            loglik = np.sum(loglik, axis=1)  # + const_array

            loglik = loglik - logsumexp_scipy(loglik)
            loglik = loglik + const_array

            c_lik = np.exp(loglik - logsumexp_scipy(loglik))

            # if np.random.random() < 0.1: print sum(c_lik[:-1])
            new_cluster_idx = np.nonzero(np.random.multinomial(1, c_lik) == 1)[0][0]
            if new_cluster_idx == self.n_clusters:  # new cluster
                mut.assigned_to -= mut
                DP_cluster(self, mut)  # create new cluster, lists updated automatically
            else:
                new_cluster = self.clusterlist[new_cluster_idx]
                mut.assigned_to -= mut
                new_cluster += mut

        cluster_counter = itertools.count()
        next(cluster_counter)
        real_index = dict([[x.id, next(cluster_counter)] for x in self.clusterlist])
        self.results.assign.append([real_index[x.assigned_to.id] for x in self.mutations])
        self.results.alpha.append(self.alpha)
        self.results.eta.append(self.eta)
        self.results.cluster_loghistograms.append([cluster.normed_hist for cluster in self.clusterlist])
        self.results.cluster_positions.append(
            [[np.argmax(x) for x in cluster.normed_hist] for cluster in self.clusterlist])
        self.results.clust_prop.append([len(cluster) / float(self.n_muts) for cluster in self.clusterlist])
        self.results.clust_size.append([len(cluster) for cluster in self.clusterlist])
        self.results.K.append(self.n_clusters)

        return [real_index[x.assigned_to.id] for x in self.mutations], [cluster.normed_hist for cluster in
                                                                        self.clusterlist]

    def tree_cluster_DP(self, co_assign, K, clonal_threshold=0.95, var_loc=None, DP_res=None, N_burn=0, tsne=True,
                        grid_size=101):

        # Flatten for TSNE and summary.
        log_data_flattened = []

        for label, mutation in self.data.iteritems():
            log_data_flattened.append(mutation.flatten())

        '''Normalize for TSNE'''

        log_data_flattened = np.array(log_data_flattened, dtype=np.longdouble)
        if tsne:
            data_norm = preprocessing.normalize(np.exp(log_data_flattened), norm='l2')

        for sample in range(self.n_samples):
            var_loc[:, 101 * sample] = 1e-40

        assign = scipy_cluster.hierarchy.fclusterdata(var_loc, K, criterion='maxclust', metric="cityblock",
                                                      method="average")
        # if tsne:
        #   logging.debug("PERPLEXITY",float(len(assign))/600.0*self.n_samples*30)
        #    before=manifold.TSNE(metric="cityblock",n_iter=20000,init="pca",perplexity=float(len(assign))/600.0*self.n_samples*30,learning_rate=200).fit_transform(data_norm)
        #    after=manifold.TSNE(metric="cityblock",n_iter=20000,init="pca",perplexity=float(len(assign))/600.0*self.n_samples*30,learning_rate=200).fit_transform(var_loc)
        # else:
        before = []
        after = []

        CL = set(assign)

        mcmc_clust_dens = []
        cluster_dens = []

        for i, new_cluster in enumerate(CL):
            log_f_post = get_norm_marg_hist(
                np.reshape(np.sum(log_data_flattened[assign == new_cluster, :], 0, dtype=np.float32),
                           (self.n_samples, self.num_bins)), self.logprior)
            cluster_dens.append(np.exp(log_f_post))

            mcmc_clust_dens.append(
                np.reshape(np.sum(var_loc[assign == new_cluster, :], axis=0), (self.n_samples, self.num_bins)))

        # Now we have to sort the cluster densities and the assignments.
        cluster_max = [sum(np.argmax(x.reshape(-1, grid_size), axis=1)) for x in cluster_dens]
        new_cluster_order = sorted(range(1, len(cluster_dens) + 1), key=lambda x: cluster_max[x - 1], reverse=True)

        assign_sorted = np.array(assign, copy=True)
        for cluster_idx_sorted, cluster_idx in enumerate(new_cluster_order):
            assign_sorted[assign == cluster_idx] = cluster_idx_sorted + 1

        _, cluster_dens_sorted = zip(*sorted(enumerate(cluster_dens), key=lambda x: cluster_max[x[0]], reverse=True))
        _, mcmc_clust_dens_sorted = zip(
            *sorted(enumerate(mcmc_clust_dens), key=lambda x: cluster_max[x[0]], reverse=True))

        # assign sorted densities back over the originals
        cluster_dens = cluster_dens_sorted
        assign = assign_sorted
        mcmc_clust_dens = mcmc_clust_dens_sorted

        return {"assign": assign, "clust_CCF_dens": cluster_dens, "tSNE": [before, after], "mcmc": mcmc_clust_dens}

    @property
    def total_grid_size(self):
        return self.num_bins * self.n_samples

    @property
    def n_clusters(self):
        return len(self.clusterlist)

    @property
    def n_muts(self):
        return len(self.mutations)

    @staticmethod
    def logsum_of_marginals(loghist):
        return np.sum(np.apply_along_axis(lambda x: logsumexp(x), 1, loghist))

    @staticmethod
    def logsum_of_marginals_per_sample(loghist):
        return np.apply_along_axis(lambda x: logsumexp(x), 1, np.array(loghist, dtype=np.float32))

    def normalize_loghist_with_prior(self, loghist):
        # Normalize in each dimension
        return np.apply_along_axis(lambda x: x - logsumexp(x), 1, loghist + self.logprior)


from scipy.interpolate import interp1d

grid = np.linspace(0, 1, 101)


def bin_x_and_calculate_delta(x_in):
    # first, bin x_in's using midpoint method
    binned_x = [x_in[j] + (x_in[j + 1] - x_in[j]) / 2. for j in range(len(x_in) - 1)]
    binned_x.insert(0, x_in[0] - (x_in[1] - x_in[0]) / 2.)
    binned_x.append(x_in[-1] + (x_in[-1] - x_in[-2]) / 2.)
    binned_x = np.array(binned_x)
    # calculate bin widths for each bin in binned_x_in
    delta_x = np.array([max(0.000001, binned_x[j + 1] - binned_x[j]) for j in range(len(binned_x) - 1)])
    return delta_x


def change_of_variables(x_in, y_in, x_out):
    ##################################################################
    # Function to do a change of variables for prob density function #
    ##################################################################
    # x_in -- list, original domain
    # y_in -- list, original pdf
    # x_out -- list, transformed domain (has to be same dim as x_in)

    # bin x_in
    delta_x_in = bin_x_and_calculate_delta(x_in)

    # calculate the integral of the function over all bins and renormalize
    y_in = np.array(y_in, dtype=np.float32)
    # y_in = y_in - logsumexp(y_in + np.log( delta_x_in,dtype=np.float32))
    # for each bin, keep integral of each bin consistent, but transform widths according to transformation
    delta_x_out = bin_x_and_calculate_delta(x_out)

    y_out = np.array(y_in + np.log(delta_x_in) - np.log(delta_x_out), dtype=np.float32)
    return y_out - logsumexp(y_out)


class DP_item:
    def __init__(self, label, engine, assignment=None, concordant_with=set(), discordant_with=set()):
        self._DP_engine_parent = engine
        self.id = label
        self._hash = hash(self.id)
        self.assigned_to = assignment
        self.item_prior = \
        self._DP_engine_parent.logsum_of_marginals_per_sample(self.loghist + self._DP_engine_parent.logprior)[0]

        ###
        self.concordant_with = set()
        self.discordant_with = set()
        ####

        self._DP_engine_parent.mutations.append(self)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other

    @property
    def loghist(self):
        return self._DP_engine_parent.data[self.id]


class DP_cluster:
    _id_generator = itertools.count()

    def __init__(self, engine, mut):
        self._DP_engine_parent = engine
        self._loghist = np.zeros((self._DP_engine_parent.n_samples, self._DP_engine_parent.num_bins), dtype=np.float32)
        self.id = next(DP_cluster._id_generator)
        self._muts = set()

        self += mut
        self._DP_engine_parent.clusterlist.append(self)

    # __hash__ and __eq__ both required for hashable class.
    def __hash__(self):
        return self.id

    def __eq__(self, other):
        # A mutation should never be in more than one cluster.
        return self._muts == other._muts

    def __len__(self):
        return len(self._muts)

    def __contains__(self, other):
        return other in self._muts

    @property
    def loghist(self):
        return self._loghist

    @property
    def normed_hist(self):
        return self._normed_loghist

    def correct_hist(self, hist):
        return hist
        new_hist = []
        for s_idx, row in enumerate(hist):
            new_hist.append(change_of_variables(self._DP_engine_parent.data._phasing[s_idx](grid), row, grid))

        return np.array(new_hist, dtype=np.float32)

    def __iadd__(self, other):
        # add mutation to cluster
        other.assigned_to = self
        self._muts.add(other)
        self._loghist += other.loghist

        if len(self) < 0:

            self._normed_loghist = self._DP_engine_parent.normalize_loghist_with_prior(self.correct_hist(self._loghist))
        else:

            self._normed_loghist = self._DP_engine_parent.normalize_loghist_with_prior(self._loghist)

        return self

    def __isub__(self, other):
        # remove mutation from cluster
        other.assigned_to = None
        self._muts.remove(other)

        if len(self._muts) > 0:
            self._loghist -= other.loghist

            if len(self) < 0:
                self._normed_loghist = self._DP_engine_parent.normalize_loghist_with_prior(
                    self.correct_hist(self._loghist))
            else:
                self._normed_loghist = self._DP_engine_parent.normalize_loghist_with_prior(self._loghist)

        else:
            # If no mutations left don't bother adding anything, just delete the cluster.
            self._DP_engine_parent.clusterlist.remove(self)

    def __add__(self, other):
        return self.correct_hist(self._loghist + other.loghist)

    def __sub__(self, other):
        return self.correct_hist(self._loghist - other.loghist)


class ClusteringResults:
    """CLASS info
    A class to store results from clustering
    PROPERTIES:
    ClusteringResults.assign: cluster id# per mutation. 1 indexed.
    ClusteringResults.clust_CCF_dens: array of cluster densities
    ClusteringResults.tSNE: TSNE visualization of clusters. [BEFORE,AFTER]
    """
    warned_read = False
    warned_write = False

    @classmethod
    def _create_from_dict(cls, legacy_dict):
        # logging.warning("!!! Creating results object from dictionary, this functionality will be removed. !!!")
        res_obj = cls()
        for key, value in legacy_dict.iteritems():
            setattr(res_obj, key, value)
        return res_obj

    def __init__(self, assign=None, clust_CCF_dens=None, tSNE=None, mcmc=None, var_loc=None):
        self.clust_CCF_dens = clust_CCF_dens
        self.tSNE = tSNE
        self.mcmc = mcmc
        self.var_loc = var_loc
        self.assign = assign

        self.cluster_loghistograms = []
        self.cluster_positions = []
        self.clust_prop = []
        self.clust_size = []

        self.K = []
        self.eta = []
        self.alpha = []
        # this is a dupilicate parameter, the overwrite is intentional. Save the assign. across samples.
        self.assign = []

        self.debug_outputs = []

    # compatability ouputs. will be removed.
    @property
    def c_fpost(self):
        return [[y.flatten() for y in x] for x in self.cluster_loghistograms]

    def __getitem__(self, key):
        if not self.warned_read:
            # logging.warning("!!! Reading results from object as dictionary, this functionailty will be removed. !!!")
            self.warned_read = True
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not self.warned_write:
            # logging.warning("!!! Writing to results object as dictionary, this functionailty will be removed. !!!")
            self.warned_write = True

        setattr(self, key, value)


def log_add_by_margs(probability_distr):
    tot = 0

    for row in probability_distr:
        tot += logsumexp(row)

    return tot


def get_norm_marg_hist(log_sum, log_prior):
    # Normalize in each dimension
    log_f_LL = log_sum + log_prior

    log_f_post = []

    for row in log_f_LL:
        log_f_post.append(row - logsumexp(row))

    return np.array(log_f_post, dtype=np.float32)


def log_sum_prod_two_hist(log_c_res, log_CCF_dist):
    # Add normalized mutation to normalized cluster in each dimension, then log_sumexp
    pr = log_c_res + log_CCF_dist

    res = log_add_by_margs(pr)

    return res


def sample_gamma_cond_N_k(N, k, eta, Pi_gamma):
    a = Pi_gamma[0]
    b = Pi_gamma[1]

    m1 = stats.gamma.rvs(a + k, scale=1. / (b - np.log(eta)))
    m2 = stats.gamma.rvs(a + k - 1, scale=1. / (b - np.log(eta)))

    D = N * (b - np.log(eta))
    w = (a + k - 1) / float((D + a + k))
    pi_eta_ratio = (a + k - 1) / (N * float(b - np.log(eta)))
    pi_eta = pi_eta_ratio / (pi_eta_ratio + 1)

    new_gamma = pi_eta * m1 + (1 - pi_eta) * m2

    return (new_gamma)


def get_gamma_prior_from_k_prior(N, k_0_map, k_prior):
    def LL(Par, N, int_k_prior):

        if (any(Par < 0)): return np.inf

        mu = Par[0]
        sigma = Par[1]
        B = mu / float(sigma ** 2)
        A = B * mu

        k_prob = stats.gamma.pdf(k_0_map[:, 1], A, scale=1.0 / B)
        k_prob = k_prob * np.append(1, 1. / np.diff(k_0_map[:, 0]))
        k_prob = k_prob / float(np.sum(k_prob))

        ix = np.logical_and(k_prob > 0, int_k_prior > 0)
        t1 = sum((np.log(k_prob) * k_prob)[ix])
        t2 = sum((np.log(int_k_prior) * k_prob)[ix])
        divergence = (t1 - t2) / float(len(int_k_prior))

        return divergence

    sigma_grid = range(1, 25, 5)
    mu_grid = range(1, 25, 5)
    obj = np.ones([len(sigma_grid), len(mu_grid)]) * np.nan
    mode_vals = np.ones([len(sigma_grid) * len(mu_grid), 3]) * np.nan

    int_k_prior = np.interp(k_0_map[:, 0], range(1, N + 1), k_prior)
    int_k_prior = int_k_prior / float(np.sum(int_k_prior))

    for i in range(len(sigma_grid)):
        for j in range(len(mu_grid)):
            par0 = [sigma_grid[i], mu_grid[j]]
            res = minimize(lambda x: LL(x, N, int_k_prior), par0, method="Nelder-Mead")

            val = res.x
            obj[i, j] = LL(val, N, int_k_prior)
            mode_vals[(i - 2) * len(mu_grid) + j, :] = np.append(val, obj[i, j])
            print
            '.',

    logging.info("")

    ix = np.argmin(mode_vals[:, 2])
    mu = mode_vals[ix, 0]
    sigma = mode_vals[ix, 1]
    KL_divergence = mode_vals[ix, 2]

    B = mu / float(sigma ** 2)
    A = B * mu
    val = {"a": A, "b": B, "KL_divergence": KL_divergence}

    return (val)


def summarize_mut_locations(DP_res, N_burn):
    cf_post = DP_res["c_fpost"]
    N_iter = len(DP_res["K"])
    assign = np.array(DP_res["assign"]).T

    N_mut = np.shape(assign)[0]
    N_GRID = np.shape(cf_post[2])[1]
    loc = np.zeros([N_GRID, N_mut])

    n_clusters = list(DP_res["K"][N_burn:])
    cluster_counts = collections.Counter(n_clusters)
    sorted_counts = np.array(list(cluster_counts.items()))
    print ('Sorted counts')
    print(sorted_counts)
    sorted_counts = sorted_counts[sorted_counts[:, 0].argsort()]
    num_cluster_index = 0

    while sorted_counts[num_cluster_index][1] * 10 < max(sorted_counts[:, 1]):
        num_cluster_index += 1
        if num_cluster_index == len(sorted_counts):
            logging.error("Cluster Number unstable, increase iterations! (--n_iter X)")
            sys.exit(1)

    K = int(sorted_counts[num_cluster_index][0])

    logging.info("N Clusters = {}".format(K))

    b = np.array(DP_res["K"])  # <= K
    indices = np.array(range(len(b)))
    indices = indices[np.logical_and(indices > N_burn, b == K)]

    for i in indices:

        CL = list(set(assign[:, i]))
        # For each unique cluster in each iteration
        for cluster_id in CL:
            c_dat = assign[:, i] == cluster_id  # Indicies of mutations equal to this cluster.

            # Each bin of every mutation in this cluster is increased by the value of this cluster at this iteration.
            loc[:, c_dat] = loc[:, c_dat] + np.tile(np.exp(cf_post[i][int(cluster_id) - 1]), [sum(c_dat), 1]).T

    res = (loc / (N_iter - N_burn + 1)).T  # normalize and transpose

    return res, K


def get_log_stirling_coefs(W):
    ## if W = c(1:N), then this function returns the (unisgned 1st kind of) Stirling numbers for
    ## N, k=c(1:N)
    ## starts to give incorrect results at N = 19 if fft is used
    ## x and y logged
    def log_conv(x, y):
        ## y is len 2
        try:
            x = [-np.inf] + list(x) + [-np.inf]

        except:

            x = [-np.inf, x, -np.inf]
        x.insert(0, -np.inf)

        res = [np.nan] * (len(x) - 1)
        for k in range(len(x) - 1):
            res[k] = logsumexp_scipy([x[k] + y[0], x[k + 1] + y[1]])  #

        return (res)

    N = len(W)
    nW = W

    cres = np.log(1)
    for i in range(N - 1):
        cres = log_conv(cres, np.log([nW[i], 1]))

    return list(reversed(cres))


def DP_prob_k_cond_alpha_N(N, alpha, log_stirling_coef):
    loglik = [np.nan] * N
    for k in range(1, N + 1):
        loglik[k - 1] = log_stirling_coef[k - 1] + lgamma(N - 1) + k * np.log(alpha) + lgamma(alpha) - lgamma(alpha + N)

    Pr = np.exp(loglik - logsumexp_scipy(loglik))

    return (Pr)


# map 1st moments of (k | N, gamma), over a grid on gamma
def get_k_0_map(N, gamma_GRID, log_stirling_coef):
    #   gamma_GRID = seq(0.01,10, length.out = 1000)
    N_gamma = len(gamma_GRID)

    k_0_map = np.zeros([N_gamma, 2])

    k_0_map[:, 1] = gamma_GRID

    DP_prob = np.zeros([N_gamma, N])
    for i in range(N_gamma):
        DP_prob[i, :] = DP_prob_k_cond_alpha_N(N, gamma_GRID[i],
                                               log_stirling_coef)  ## Escobar and West 1995 for DP loglik eqn 10.
        k_0_map[i, 0] = np.sum(DP_prob[i, :] * np.array(range(1, N + 1)))

    return (k_0_map)


def init_dp_prior(N, Pi_k):
    k_prior = stats.nbinom.pmf(range(1, N + 1), Pi_k["r"], Pi_k["r"] / float(Pi_k["r"] + Pi_k[
        "mu"]))  # stats.nbinom.pmf(range(1,N+1),Pi_k["mu"]+ Pi_k["mu"]**2/float(Pi_k["r"]), Pi_k["r"]/float(Pi_k["r"]+Pi_k["mu"]) )

    k_prior = k_prior / sum(k_prior)

    logging.info("Initializing prior over DP k for " + str(N) + " items")

    log_stirling_coef = get_log_stirling_coefs(range(1, N + 1))

    GMAX = 5
    grid = np.linspace(1e-25, GMAX, 1000)
    k_0_map = get_k_0_map(N, grid, log_stirling_coef)

    Pi_gamma = get_gamma_prior_from_k_prior(N, k_0_map, k_prior)

    return Pi_gamma

import logging
import numpy as np
from scipy.special import logsumexp as logsumexp_scipy

import pkgutil

if pkgutil.find_loader('sselogsumexp') is not None:
    logging.info("Using fast logsumexp")
    from sselogsumexp import logsumexp
else:
    logging.info("Using scipy (slower) logsumexp")
    from scipy.misc import logsumexp


class Cluster:

    def __init__(self, identifier, time_points, num_bins=101, blacklist_threshold=0.1):
        self._identifier = identifier
        self._blacklist_threshold = blacklist_threshold
        self._densities = {}
        # Dictionary of mutations with key: mutation.var_str and value: nd histogram in log space
        self._mutations = {}
        self._time_points = time_points
        self._num_bins = num_bins
        self._hist = np.zeros((len(self._time_points), self._num_bins), dtype=np.float32)
        self._loghist = np.zeros((len(self._time_points), self._num_bins), dtype=np.float32)
        self._logprior = np.zeros((len(self._time_points), self._num_bins), dtype=np.float32)
        # Blacklists cluster that have low ccf mean across all samples
        self._blacklisted = None
        # For each iteration of MCMC record number of mutations added and removed
        self._iter_count_removed = 0
        self._iter_count_added = 0

    def reset_counts(self):
        # Reset iteration counts for each iteration of MCMC
        self._iter_count_removed = 0
        self._iter_count_added = 0

    def __eq__(self, other):
        """ """
        if other is None:
            return False
        # TODO decide if the same identifiers are also required for equality
        if self._mutations == other.mutations and self._ccf == other.ccf:
            return True
        else:
            return False

    @property
    def identifier(self):
        """ Cluster ID """
        return self._identifier

    @property
    def blacklisted(self):
        """ Blacklist status of the cluster """
        return self._blacklisted

    @property
    def densities(self):
        """ Mapping between sample name and histogram """
        return self._densities

    @property
    def hist(self):
        """ Normalized density histogram (not log space) """
        return self._hist

    @property
    def loghist(self):
        """ Log space normalized density histogram """
        return self._loghist

    @property
    def logprior(self):
        """ Density distribution output of clustering in log-space """
        return self._logprior

    @property
    def mutations(self):
        """ Dictionary of mutations that are assigned to this cluster """
        return self._mutations

    @property
    def cluster_size(self):
        """ Number of mutations in the cluster """
        return len(self._mutations)

    # (mut, mut_nd_hist, update_cluster_hist=False, create_mut_nd_hist=False)
    def add_mutation(self, mutation, mutation_nd_hist, update_cluster_hist=False, create_mut_nd_hist=False):
        """ Initially when mutations are added they shouldn't change cluster density
            When reshuffling mutations cluster density should be updated """
        self._iter_count_added += 1
        if mutation not in self._mutations:
            if create_mut_nd_hist:
                mutation_nd_hist = self._make_nd_histogram(mutation_nd_hist)
            self._mutations[mutation] = mutation_nd_hist
            if update_cluster_hist:
                self._update_hist(mutation_nd_hist, action='add')
            logging.debug("Added mutation {} to cluster {}.".format(mutation.var_str, self._identifier))
        else:
            logging.error(
                "Can not add mutation {} to cluster {}. It is already there.".format(mutation.var_str,
                                                                                     self._identifier))

    def remove_mutation(self, mutation, update_cluster_hist=True):
        """ """
        self._iter_count_removed += 1
        if mutation in self._mutations:
            mutation_nd_hist = self._mutations[mutation]
            del self._mutations[mutation]
            if update_cluster_hist:
                self._update_hist(mutation_nd_hist, action='sub')
            logging.debug("Removed mutation {} from cluster {}.".format(mutation.var_str, self._identifier))
        else:
            logging.error("Can not remove mutation {} from cluster {}. It is not there".format(mutation.var_str,
                                                                                               self._identifier))

    @staticmethod
    def _make_nd_histogram(hist_array, conv=1e-40):
        hist = np.asarray(hist_array, dtype=np.float32) + conv
        return np.apply_along_axis(lambda z: z - logsumexp_scipy(z), 1, np.log(hist))

    def _normalize_loghist_with_prior(self, loghist):
        """ Normalize in each dimension in log space """
        loghist = np.asarray(loghist, dtype=np.float32)
        return np.apply_along_axis(lambda x: x - logsumexp(x), 1, loghist + self._logprior)

    def _add_mutation(self, mut_hist):
        """ Update cluster density after adding mutation to it """
        # TODO: check if it is an empty cluster
        self._loghist = self._loghist + mut_hist

    def _sub_mutation(self, mut_hist):
        """ Update cluster density after removing mutation from it """
        # TODO: what if it is the last mutation in cluster
        self._loghist = self._loghist - mut_hist

    def _update_hist(self, mut_hist, action='add'):
        """ Updating density distribution after adding or removing a mutation to/from cluster """
        if action == 'add':
            self._add_mutation(mut_hist)
        elif action == 'sub':
            self._sub_mutation(mut_hist)
        self._loghist = self._normalize_loghist_with_prior(self._loghist)
        # Updating hist
        logging.debug('Updating histogram after adding or removing a mutation')
        self._hist = np.apply_along_axis(np.exp, 1, self._loghist)

    def add_mutations(self, mutations):
        for mutation in mutations:
            self.add_mutation(mutation)

    def add_sample_density(self, sample_id, density, conv=1e-40):
        sample_idx = self._time_points.index(sample_id)
        density = np.asarray(density, dtype=np.float32) + conv
        log_density = np.log(density, dtype=np.float32)
        self._hist[sample_idx] = density
        self._logprior[sample_idx] = log_density - logsumexp_scipy(log_density)
        self._loghist[sample_idx] = log_density - logsumexp_scipy(log_density)
        logging.debug('Added density for cluster {} for sample {}'.format(self._identifier, sample_id))

    def cluster_means(self):
        grid_size = self._hist.shape[1]
        return np.sum(self._hist * np.arange(grid_size) / (grid_size - 1), axis=1)

    def _low_ccf_check(self):
        return all([ccf_mean <= self._blacklist_threshold for ccf_mean in self.cluster_means()])

    def set_blacklist_status(self):
        """ Checks if ccf mean across all samples is below a threshold, blacklist cluster """
        if self._low_ccf_check():
            self._blacklisted = True
        else:
            self._blacklisted = False

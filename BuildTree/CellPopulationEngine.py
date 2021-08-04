from scipy import stats
import random
from random import shuffle
from collections import defaultdict
import collections
import numpy as np
import itertools
import operator
import logging


class CellPopulationEngine:

    def __init__(self, patient, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed=seed)
        self._patient = patient
        self._all_configurations = {}             
        if patient.ClusteringResults:
            self._clustering_results = patient.ClusteringResults
        else:
            logging.error('Clustering results are not found, need to run Clustering module prior to Cell Population')
        if patient.TopTree:
            self._top_tree = patient.TopTree
            logging.debug('Loaded top tree with edges {}'.format(self._top_tree.edges))
        else:
            logging.error('Build Tree results are not found, need to run BuildTree prior to running Cell Population')

    def get_random_cluster(self):
        return np.random.choice(self._clusters)

    @property
    def mcmc_trace(self):
        return self._all_configurations

    @staticmethod
    def sample_ccf(xk, pk):
        """ """
        if sum(pk) == 0:
            return 0
        else:
            pk = pk / sum(pk)
            custm = stats.rv_discrete(name='custm', values=(xk, pk))
            return custm.rvs(size=1)[0]

    @staticmethod
    def logSumExp(ns):
        max_ = np.max(ns)
        ds = ns - max_
        sumOfExp = np.exp(ds).sum()
        return max_ + np.log(sumOfExp)

    def _normalize_in_logspace(self, dist, in_log_space=True):
        if in_log_space:
            log_dist = np.array(dist, dtype=np.float32)
        else:
            log_dist = np.log(dist, dtype=np.float32)
        return np.exp(log_dist - self.logSumExp(log_dist))

    def _compute_cluster_constrained_density(self, node, cluster_density, sample_id, iter_ccf, conv=1e-40):
        parent = node.parent
        if parent:
            # it is required that parent constrained ccf was already assigned
            if parent.data.identifier in iter_ccf:
                parent_cluster_ccf = iter_ccf[parent.data.identifier]
                siblings_total_ccf = sum([iter_ccf[sibling] for sibling in node.siblings if sibling in iter_ccf])
                leftover_ccf = int(parent_cluster_ccf - siblings_total_ccf)                
                cluster_constrained_density = list(cluster_density[:leftover_ccf + 1]) + [0.0] * (100 - leftover_ccf)
                cluster_constrained_density[leftover_ccf] += sum(cluster_density[leftover_ccf+1:])
                if sum(cluster_constrained_density) > 0:
                    return self._normalize_in_logspace(cluster_constrained_density, in_log_space=False)
                else:
                    return None
            else:
                logging.error('Parent {} ccf was not assigned'.format(parent.identifier))
                return None
        else:
            return cluster_density

    def sample_cluster_ccf(self, node, sample_clusters_ccf, sample_id, iter_ccf, hist):
        cluster_ccf = sample_clusters_ccf[node.data.identifier]
        cluster_constrained_density = self._compute_cluster_constrained_density(node, cluster_ccf, sample_id, iter_ccf)                      
        if cluster_constrained_density is not None:            
            cluster_constrained_density = self._normalize_in_logspace(cluster_constrained_density, in_log_space=False)
            return self.sample_ccf(hist, cluster_constrained_density)
        else:            
            logging.warn('Constrained ccf for node {} is None'.format(node.identifier))
            return 0.0        

    def _compute_sample_constrained_ccf(self, sample_clusters_ccf, sample_id, tree_levels, n_iter=250, burn_in=100):
        """ For each cluster computes constrained density and samples ccf from that density
            :returns the most frequent ccf guration for the sample across all iterations """
        hist = range(101)
        sample_mcmc_trace = []
        for i in range(n_iter + burn_in):
            logging.debug('Iteration {}'.format(i))
            iter_ccf = dict.fromkeys(itertools.chain(self._top_tree.nodes.keys()), 0.0)
            # Traverse tree from root to it's leaves
            for level in tree_levels:
                level_nodes = tree_levels[level]
                shuffle(level_nodes)
                # For each node in the level
                for node_id in level_nodes:
                    node = self._top_tree.nodes[node_id]
                    cluster_sampled_ccf = self.sample_cluster_ccf(node, sample_clusters_ccf, sample_id, iter_ccf, hist)
                    logging.debug('Cluster {} has constrained ccf {}'.format(node_id, cluster_sampled_ccf))
                    iter_ccf[node.data.identifier] = cluster_sampled_ccf
            if i >= burn_in:
                sample_mcmc_trace.append(iter_ccf)
        return sample_mcmc_trace

    def _get_sample_clusters_densities(self, sample_id):
        """ For each sample ID returns dictionary of clusters and their densities for this sample """
        sample_clusters_densities = {}
        for cluster_id, cluster in self._clustering_results.items():
            sample_clusters_densities[cluster_id] = cluster.hist[sample_id]
        return sample_clusters_densities

    def get_all_constrained_ccfs(self):
        """ For each MCMC iteration returns constrained ccfs """
        return self._all_configurations
        # all_cell_ccfs = {}
        # for sample_id, sample_constrained_ccf in self._all_configurations.items():
        #     all_cell_ccfs[sample_id] = []
        #     for iteration, constrained_ccf in enumerate(sample_constrained_ccf):
        #         all_cell_ccfs[sample_id].append(constrained_ccf)
        # return all_cell_ccfs

    def get_all_cell_abundances(self):
        """ For each MCMC iteration computes cell abundances """
        all_cell_abundances = {}
        for sample_id, sample_constrained_ccf in self._all_configurations.items():
            all_cell_abundances[sample_id] = []
            for iteration, constrained_ccf in enumerate(sample_constrained_ccf):
                all_cell_abundances[sample_id].append(self.get_cell_abundance(constrained_ccf))
        return all_cell_abundances

    def get_cell_abundance_across_samples(self, constrained_ccf):
        """ For each sample computes cell abundances for each cluster """
        cell_abundances_across_samples = {}
        for sample_id, sample_constrained_ccf in constrained_ccf.items():
            sample_cell_abundance = self.get_cell_abundance(sample_constrained_ccf)
            cell_abundances_across_samples[sample_id] = sample_cell_abundance
            logging.debug('Cell abundance for sample {} \n {}'.format(sample_id, sample_cell_abundance))
        return cell_abundances_across_samples

    def get_cell_abundance(self, sample_constrained_ccfs):
        """ Adjusts constrained ccf to represent cell population according to phylogenetic tree """
        # Have to convert to dictionary, in case list of tuples is passed
        sample_constrained_ccfs = dict(sample_constrained_ccfs)
        sample_cell_abundance = {key: 0 for key, value in sample_constrained_ccfs.items()}
        for node_id, node_ccf in sample_constrained_ccfs.items():
            node = self._top_tree.nodes[node_id]
            parent = node.parent
            if parent:
                logging.debug('Node {} has parent {} with abundance {}'.format(node_id, parent.identifier,
                                                                               sample_cell_abundance[
                                                                                   parent.identifier]))
                sample_cell_abundance[parent.identifier] -= node_ccf
            else:
                logging.debug('Node {} has no parent'.format(node_id))
            sample_cell_abundance[node_id] += node_ccf
        # check that cancer cell population in the sample sums up to 100%
        assert sum([a for cl, a in sample_cell_abundance.items()]) <= 100.0
        return sample_cell_abundance

    @staticmethod
    def _get_most_frequent_configuration(constrained_ccfs_configs):
        sorted_config = []
        for d in constrained_ccfs_configs:
            sorted_config.append(tuple(sorted(d.items(), key=operator.itemgetter(1))))
        counts = collections.Counter(sorted_config)
        most_frequent = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[0]
        return sorted(most_frequent[0], key=operator.itemgetter(0)), most_frequent[1]

    def compute_constrained_ccf(self, n_iter=250):
        """ For each sample, iterates over tree levels and computes average ccf for each cluster
            in that level according to phylogenetic tree """
        tree_levels = self._top_tree.get_tree_levels()
        samples_ccf = {}
        for idx, sample in enumerate(self._patient.sample_list):
            sample_clusters_density = self._get_sample_clusters_densities(idx)
            sample_mcmc_trace = self._compute_sample_constrained_ccf(sample_clusters_density, sample.sample_name, tree_levels, n_iter)
            most_frequent_config, most_frequent_count = self._get_most_frequent_configuration(sample_mcmc_trace)
            samples_ccf[sample.sample_name] = most_frequent_config
            # For each sample record its MCMC trace
            self._all_configurations[sample.sample_name] = sample_mcmc_trace
            logging.debug('Most frequent constrained ccf configuration with count {} \n{} '.format(most_frequent_count,
                                                                                                   most_frequent_config))
        return samples_ccf

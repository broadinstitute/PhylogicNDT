from scipy import stats
from random import shuffle
import collections
import numpy as np
import itertools
import operator
import logging
from scipy.special import logsumexp as logsumexp_scipy


class CellPopulationEngine:

    def __init__(self, patient):
        self._patient = patient
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

    @staticmethod
    def sample_ccf(xk, pk):
        """ """
        if sum(pk) == 0:
            return 0
        else:
            custm = stats.rv_discrete(name='custm', values=(xk, pk))
            return custm.rvs(size=1)[0]

    @staticmethod
    def _normalize_in_logspace(dist, in_log_space=True):
        logging.debug('Distribution before normalization\n{}'.format(dist))
        if in_log_space:
            log_dist = np.array(dist, dtype=np.float64)
        else:
            logging.debug('Converting distribution to log space before normalization')
            log_dist = np.log(dist, dtype=np.float64)
        return np.exp(log_dist - logsumexp_scipy(log_dist))

    def _compute_cluster_constrained_density(self, node, cluster_density, iter_ccf):
        parent = node.parent
        if parent:
            if parent.data.identifier in iter_ccf:
                parent_cluster_ccf = iter_ccf[parent.data.identifier]
                siblings_total_ccf = sum([iter_ccf[sibling] for sibling in node.siblings if sibling in iter_ccf])
                leftover_ccf = int(parent_cluster_ccf - siblings_total_ccf)
                logging.debug('Node {} has parent {} with ccf {}'.format(node.identifier,
                                                                         node.parent.identifier,
                                                                         parent_cluster_ccf))
                logging.debug('Node {} has siblings {} with total ccf {}'.format(node.identifier,
                                                                                 node.siblings,
                                                                                 siblings_total_ccf))
                logging.debug('Node {} has leftover ccf {}'.format(node.identifier,
                                                                   leftover_ccf))

                constrained_cluster_density = list(cluster_density[:leftover_ccf + 1]) + [0.0] * (100 - leftover_ccf)
                if sum(constrained_cluster_density) > 0:
                    return self._normalize_in_logspace(constrained_cluster_density, in_log_space=False)
                else:
                    return None
            else:
                logging.error('Parent {} ccf was not assigned'.format(parent.identifier))
                return None
        else:
            return cluster_density

    def sample_cluster_ccf(self, node, sample_clusters_ccf, iter_ccf, hist):
        cluster_ccf = sample_clusters_ccf[node.data.identifier]
        cluster_constrained_density = self._compute_cluster_constrained_density(node, cluster_ccf, iter_ccf)
        logging.debug(
            'Constrained distribution for node {} is \n{}'.format(node.identifier, cluster_constrained_density))
        if cluster_constrained_density is not None:
            return self.sample_ccf(hist, cluster_constrained_density)
        else:
            logging.warn('Constrained ccf for node {} is None'.format(node.identifier))
            return 0.0

    def _compute_sample_constrained_ccf(self, sample_clusters_ccf, tree_levels, n_iter=250):
        """ For each cluster computes constrained density and samples ccf from that density
            :returns the most frequent ccf configuration for the sample across all iterations """
        hist = range(101)
        all_configurations = []
        for i in range(n_iter):
            logging.debug('Iteration {}'.format(i))
            iter_ccf = dict.fromkeys(itertools.chain(self._top_tree.nodes.keys()), 0.0)
            # Traverse tree from root to it's leaves
            for level in tree_levels:
                level_nodes = tree_levels[level]
                shuffle(level_nodes)
                # For each node in the level
                for node_id in level_nodes:
                    node = self._top_tree.nodes[node_id]
                    cluster_sampled_ccf = self.sample_cluster_ccf(node, sample_clusters_ccf, iter_ccf, hist)
                    logging.debug('Cluster {} has constrained ccf {}'.format(node_id, cluster_sampled_ccf))
                    iter_ccf[node.data.identifier] = cluster_sampled_ccf
                all_configurations.append(iter_ccf)
        most_frequent_config, most_frequent_count = self._get_most_frequent_configuration(all_configurations)
        logging.debug('Most frequent constrained ccf configuration with count {} \n{} '.format(most_frequent_count,
                                                                                               most_frequent_config))
        return most_frequent_config

    def _get_sample_clusters_densities(self, sample_id):
        """ For each sample ID returns dictionary of clusters and their densities for this sample """
        sample_clusters_densities = {}
        for cluster_id, cluster in self._clustering_results.items():
            sample_clusters_densities[cluster_id] = cluster.hist[sample_id]
        return sample_clusters_densities

    def get_cell_abundance(self, constrained_ccf):
        """ """
        cell_abundances = {}
        for sample_id, clusters_constrained_ccf in constrained_ccf.items():
            sample_cell_abundance = {key: 0 for key, value in clusters_constrained_ccf}
            for node_id, node_ccf in clusters_constrained_ccf:
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
            cell_abundances[sample_id] = sample_cell_abundance
            logging.debug('Cell abundance for sample {} \n {}'.format(sample_id, sample_cell_abundance))
        return cell_abundances

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
            samples_ccf[sample.sample_name] = self._compute_sample_constrained_ccf(sample_clusters_density, tree_levels,
                                                                                   n_iter)
        return samples_ccf

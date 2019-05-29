import networkx as nx
from scipy import stats
from random import shuffle
import numpy as np
import itertools
import logging
# add as command line parameter
np.random.seed()

logging.basicConfig(filename='cell_population_engine.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=getattr(logging, "DEBUG"))


class CellPopulationEngine:

    def __init__(self, patient):
        self._patient = patient
        if patient.ClusteringResults:
            self._clustering_results = patient.ClusteringResults
        else:
            logging.error('Clustering results are not found, need to run Clustering module prior to Cell Population')
        if patient.TopTree:
            self._top_tree = patient.TopTree
        else:
            logging.error('Build Tree results are not found, need to run BuildTree prior to running Cell Population')

    def get_random_cluster(self):
        return np.random.choice(self._clusters)

    @staticmethod
    def sample_ccf(xk, pk):
        """

        :param xk:
        :param pk:
        :return:
        """
        if sum(pk) == 0:
            return 0
        else:
            custm = stats.rv_discrete(name='custm', values=(xk, pk))
            return custm.rvs(size=1)[0]

    @staticmethod
    def normalized(distribution):
        """

        :param distribution:
        :return:
        """
        total_sum = float(sum(distribution))
        if total_sum > 0.0:
            return [d/total_sum for d in distribution]
        else:
            return distribution

    def compute_constrained_ccf(self, node, cluster_ccf, cell_abundances, hist):
        parent = node.parent
        if parent:
            if parent.identifier in cell_abundances:
                parent_abundance = cell_abundances[parent.identifier]
                for sibling in node.siblings:
                    if sibling in cell_abundances:
                        parent_abundance -= cell_abundances[sibling]
                constrained_ccf = np.resize(cluster_ccf[:int(parent_abundance)+1], np.shape(hist))
                return self.normalized(constrained_ccf)
            else:
                logging.error('Parent {} ccf was not assigned'.format(parent.identifier))
                return None
        else:
            return cluster_ccf

    def compute_node_ccf(self, node, cluster_ccf, iter_ccf, hist):
        constrained_cluster_ccf = self.compute_constrained_ccf(node, cluster_ccf, iter_ccf, hist)
        if constrained_cluster_ccf is not None:
            return self.sample_ccf(hist, constrained_cluster_ccf)
        else:
            logging.warn('Constrained ccf for node {} is None'.format(node.identifier))
            return 0.0

    def sample_average_cell_abundance(self, sample_clusters_ccf, tree_levels, n_iter=1000):
        '''

        Args:
            sample_clusters_ccf:
            tree_levels:
            n_iter:
        Returns:
        '''
        hist = range(101)
        cell_abundances = {key: [] for key in self._top_tree.nodes.keys()}
        for i in range(n_iter):
            iter_ccf = dict.fromkeys(itertools.chain(self._top_tree.nodes.keys()), 0.0)
            # Traverse tree from root to it's leaves
            for level in tree_levels:
                level_nodes = tree_levels[level]
                shuffle(level_nodes)
                # For each node in the level
                for node_id in level_nodes:
                    node = self._top_tree.nodes[node_id]
                    node_ccf = self.compute_node_ccf(node, sample_clusters_ccf[node_id], iter_ccf, hist)
                    iter_ccf[node_id] = node_ccf
                    cell_abundances[node_id].append(node_ccf)
        return {node: int(sum(abundances) / float(len(abundances))) for node, abundances in cell_abundances.items()}

    def samples_average_constrained_ccf(self, n_iter=10):
        tree_levels = self._top_tree.get_tree_levels()
        logging.debug('Loaded top tree with edges {}'.format(self._top_tree.edges))

        clusters_ccf = self._clustering_results.clusters
        sample_names = self._clustering_results.samples
        # iterate over samples to get average cell abundance in each sample
        sample_cell_abundance = {sample_name: [] for sample_name in sample_names}
        for sample in sample_names:
            sample_clusters_ccf = {}
            for c in clusters_ccf:
                sample_clusters_ccf[c] = clusters_ccf[c].densities[sample]
            sample_cell_abundance[sample] = self.sample_average_cell_abundance(sample_clusters_ccf, tree_levels, n_iter)
        logging.debug('Average constrained ccf per sample \n{}'.format(sample_cell_abundance))
        return sample_cell_abundance

    @staticmethod
    def compute_cell_abundance(constrained_ccf, cell_ancestry):
        cell_abundances = {}
        for sample_id, cluster_constrained_ccf in constrained_ccf.items():
            cell_abundances[sample_id] = {}
            for cluster_id, ccf in cluster_constrained_ccf.items():
                if cluster_id not in cell_abundances:
                    cell_abundances[cluster_id] = ccf
                    for ancestor in cell_ancestry[cluster_id]:
                        if ancestor != cluster_id:
                            if ancestor in cell_abundances:
                                cell_abundances[ancestor] -= ccf
        logging.debug('Average cell abundance per sample \n {}'.format(cell_abundances))
        return cell_abundances





import numpy as np
import logging
import operator
import collections

from .Tree import Tree
import ShuffleMutations


class BuildTreeEngine:

    def __init__(self, patient):
        """ """
        # patient object should have pointer to clustering results object but for now it is two separate objects
        self._patient = patient
        if patient.ClusteringResults:
            self._clustering_results = patient.ClusteringResults
        else:
            logging.error(' Clustering results are not found, need to run clustering first')
        # List of likelihoods for trees at each iteration
        self._ll_trail = []
        # List of trees (list of edges) at each iteration
        self._mcmc_trace = []
        self._top_tree = None

    @property
    def trees_ll(self):
        return self._ll_trail

    @property
    def trees(self):
        return self._mcmc_trace

    @property
    def mcmc_trace(self):
        sorted_edges = map(set, sorted(self._mcmc_trace, key=lambda tup: tup[0]))
        counts = collections.Counter(map(tuple, sorted_edges))
        return sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

    def build_tree(self, n_iter=250, burn_in=100):
        """ Main function to construct phylogenetic tree """
        tree = Tree()
        tree.init_tree_from_clustering(self._patient.ClusteringResults)
        # Create initial tree, where every cluster is child of clonal

        # ND Histogram for shuffling mutations
        nd_hist = self._patient.make_ND_histogram()
        time_points = self._patient.sample_list
        for n in range(n_iter + burn_in):
            # check that it is not None
            if n <= burn_in:
                logging.debug('Burn-in iteration {}'.format(n))
            else:
                logging.debug('Iteration number {}'.format(n))
            # Randomly pick any node to move (except root, which is clonal)
            node_to_move = tree.get_random_node()
            logging.debug('Node to move {}'.format(node_to_move.identifier))
            # List of all possible Trees and corresponding likelihoods
            tree_choices, tree_choice_lik = tree.get_all_possible_moves(node_to_move, time_points)
            tree_idx = np.argmax(np.random.multinomial(1, tree_choice_lik))
            logging.debug('np.argmax(np.random.multinomial(1, tree_choice_lik)) = {}'.format(tree_idx))
            tree_edges_selected = tree_choices[tree_idx]
            if n > burn_in:
                self._mcmc_trace.append(tree_edges_selected)
                self._ll_trail.append(tree_choice_lik[tree_idx])
            logging.debug('Tree to choose edges \n{}'.format(tree_edges_selected))
            # Initialize Tree of choice for the next iteration
            # update Nodes pointers to parents and children
            tree.set_new_edges(tree_edges_selected)
            # Shuffle mutations
            ShuffleMutations.shuffling(self._patient.ClusteringResults, self._patient.sample_list)
            # Identify cluster label switching after mutation shuffling
            ShuffleMutations.fix_cluster_lables(self._patient.ClusteringResults)
        self._set_top_tree(tree)

    def _set_top_tree(self, tree):
        """ Pick the most often occurring tree """
        top_tree_edges = list(self.mcmc_trace[0][0])
        top_tree_edges.sort()
        tree.set_new_edges(top_tree_edges)
        self._top_tree = tree

    @property
    def top_tree(self):
        return self._top_tree

    def get_cell_ancestry(self):
        cells_ancestry = {}
        for node_id in self._top_tree.nodes:
            cells_ancestry[node_id] = self._top_tree.get_ancestry(node_id)
        return cells_ancestry






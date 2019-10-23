import numpy as np
import logging
import collections
import operator
import pandas as pd

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
        """
        For each iteration of MCMC counts number of times Tree appeared and computes average likelihood for it
        :return: Pandas dataframe (number of times this Tree appeared, average likelihood and list of edges)
        """
        counts = {edges: 0 for edges in set(self._mcmc_trace)}
        averg_likelihood = {edges: [] for edges in set(self._mcmc_trace)}
        merged = zip(self._mcmc_trace, self._ll_trail)
        for (edges, likelihood) in merged:
            counts[edges] += 1
            averg_likelihood[edges].append(likelihood)
        averg_likelihood = {edges: sum(likelihoods) / float(len(likelihoods)) for edges, likelihoods in
                            averg_likelihood.items()}

        sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        data = []
        for (edges, count) in sorted_counts:
            data.append(dict(
                n_iter=count,
                average_likelihood=averg_likelihood[edges],
                edges=edges
            ))
        df = pd.DataFrame(data)
        return df[['n_iter', 'average_likelihood', 'edges']]

    def build_tree(self, n_iter=250, burn_in=100):
        """ Main function to construct phylogenetic tree """
        tree = Tree()
        # Create initial tree, where each cluster is a child of the clonal cluster
        tree.init_tree_from_clustering(self._patient.ClusteringResults)
        # If tree has only one cluster, return initial Tree
        if tree.size() > 1:
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
                    self._mcmc_trace.append(tuple(tree_edges_selected))
                    self._ll_trail.append(tree_choice_lik[tree_idx])
                logging.debug('Tree to choose edges \n{}'.format(tree_edges_selected))
                # Initialize Tree of choice for the next iteration
                # update Nodes pointers to parents and children
                tree.set_new_edges(tree_edges_selected)
                # Shuffle mutations
                ShuffleMutations.shuffling(self._patient.ClusteringResults, self._patient.sample_list)
                # Identify cluster label switching after mutation shuffling
                ShuffleMutations.fix_cluster_lables(self._patient.ClusteringResults)
            top_tree_edges = self._most_common_tree()
            tree.set_new_edges(top_tree_edges)
        else:
            self._mcmc_trace.append(tuple())
            self._ll_trail.append(1)
        self.set_top_tree(tree)

    def _most_common_tree(self):
        """ Pick the most often occurring tree """
        sorted_edges = map(set, sorted(self._mcmc_trace, key=lambda tup: tup[0]))
        counts = collections.Counter(map(tuple, sorted_edges))
        return sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[0][0]

    def set_top_tree(self, tree):
        self._top_tree = tree

    @property
    def top_tree(self):
        return self._top_tree

    def get_cell_ancestry(self):
        """
        For each node in the tree returns list of all its ancestors
        :return: Dictionary (node: list of ancestors)
        """
        cells_ancestry = {}
        for node_id in self._top_tree.nodes:
            cells_ancestry[node_id] = self._top_tree.get_ancestry(node_id)
        return cells_ancestry

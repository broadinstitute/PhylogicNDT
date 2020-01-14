import numpy as np
import logging

from .Tree import Tree
from .ShuffleMutations import shuffling


class BuildTreeEngine:

    def __init__(self, patient, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        """ """
        # patient object should have pointer to clustering results object but for now it is two separate objects
        self._patient = patient
        if patient.ClusteringResults:
            self._clustering_results = patient.ClusteringResults
        else:
            logging.error(' Clustering results are not found, need to run clustering first')
        # List of likelihoods for trees at each iteration
        # self._ll_trail = []
        # List of trees (list of edges) at each iteration
        self._mcmc_trace = []
        self._top_tree = None

    def set_top_tree(self, tree):
        self._top_tree = tree

    @property
    def top_tree(self):
        return self._top_tree

    @staticmethod
    def _add_dictionary(old_dict, new_dict):
        """ Merge list of dictionaries with repeated keys
            where key - cluster id and value is list of cluster densities """
        for key, value in new_dict.items():
            if key in old_dict:
                old_dict[key].extend(value)
            else:
                old_dict[key] = value
        return old_dict

    @staticmethod
    def get_average_clusters_densities(clusters_densities, conv=1e-40):
        """ Averages and normalizes clusters densities across MCMC iterations with similar Trees"""
        from sklearn.preprocessing import normalize
        average_clusters_densities = {}
        for cluster_id, cluster_ccf_list in clusters_densities.items():
            average_density = sum(cluster_ccf_list) / float(len(cluster_ccf_list)) + conv
            normalized_average_distribution = average_density
            average_clusters_densities[cluster_id] = normalized_average_distribution
        return average_clusters_densities

    def _most_common_tree(self):
        """ Pick the most often occurring tree """
        max_iter = 0
        most_occuring_edges = None
        cluster_densities = None
        for d in self._mcmc_trace:
            if d['n_iter'] > max_iter:
                max_iter = d['n_iter']
                most_occuring_edges = d['edges']
                cluster_densities = d['cluster_densities']
        return most_occuring_edges, self.get_average_clusters_densities(cluster_densities)

    @property
    def mcmc_trace(self):
        # TODO figure out how to convert to DataFrame and sort rows according to count column
        return self._mcmc_trace

    def _update_mcmc_trace(self, edges, likelihood, clusters_densities):
        """ After each iteration of MCMC records likelihood of the tree edges, count, clusters densities """
        edges = tuple(sorted(edges, key=lambda tup: tup))
        found = False
        for d in self._mcmc_trace:
            if d['edges'] == edges:
                d['n_iter'] += 1
                d['likelihood'].append(likelihood)
                d['cluster_densities'] = self._add_dictionary(d['cluster_densities'], clusters_densities)
                found = True
                break
        if not found:
            self._mcmc_trace.append(dict(
                edges=edges,
                n_iter=1,
                likelihood=[likelihood],
                cluster_densities=clusters_densities
            ))

    def _collect_cluster_densities(self):
        # TODO: check that cluster densities are not changing
        cluster_densities = {}
        for cluster_id, cluster in self._patient.ClusteringResults.items():
            cluster_densities[cluster_id] = [cluster.hist]
        return cluster_densities

        # return {cluster_id: cluster.hist for cluster_id, cluster in self._patient.ClusteringResults.items()}

    def build_tree(self, n_iter=250, burn_in=100):
        """ Main function to construct phylogenetic tree """
        tree = Tree()
        # Create initial tree, where each cluster is a child of the clonal cluster
        tree.init_tree_from_clustering(self._patient.ClusteringResults)
        # If tree has only one cluster, return initial Tree
        if tree.size() > 1:
            time_points = self._patient.sample_list
            for n in range(n_iter + burn_in):
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
                    # Record MCMC trace for this iteration
                    self._update_mcmc_trace(tree_edges_selected, tree_choice_lik[tree_idx],
                                            self._collect_cluster_densities())
                logging.debug('Tree to choose edges \n{}'.format(tree_edges_selected))
                # Initialize Tree of choice for the next iteration
                # update Nodes pointers to parents and children
                tree.set_new_edges(tree_edges_selected)
                # Shuffle mutations
                shuffling(self._patient.ClusteringResults, self._patient.sample_list)
            top_tree_edges, cluster_densities = self._most_common_tree()
            for cluster_id, cluster in self._patient.ClusteringResults.items():
                cluster.set_hist(cluster_densities[cluster_id])
            tree.set_new_edges(top_tree_edges)
        else:
            # TODO: check that it works when there is only clonal cluster
            self._update_mcmc_trace(tuple(), 0.0, self._collect_cluster_densities())
        self.set_top_tree(tree)

    def get_cell_ancestry(self):
        """
        For each node in the tree returns list of all its ancestors
        :return: Dictionary (node: list of ancestors)
        """
        cells_ancestry = {}
        for node_id in self._top_tree.nodes:
            cells_ancestry[node_id] = ([node_id] + list(self._top_tree.get_ancestry(node_id)))[::-1]
        return cells_ancestry

    """
    @property
    def mcmc_trace(self):
        
        #For each iteration of MCMC counts number of times Tree appeared and computes average likelihood for it
        #:return: Pandas dataframe (number of times this Tree appeared, average likelihood and list of edges)
        
        tree_edges = [tuple(sorted(edges, key=lambda tup: tup)) for edges in self._mcmc_trace]
        merged = zip(tree_edges, self._ll_trail)
        averg_likelihood = {}
        counts = {}
        for edges, likelihood in merged:
            if edges in counts:
                counts[edges] += 1
                averg_likelihood[edges].append(likelihood)
            else:
                counts[edges] = 1
                averg_likelihood[edges] = [likelihood]

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
        
    def _most_common_tree(self):
        # Pick the most often occurring tree 
        sorted_edges = map(set, sorted(self._mcmc_trace, key=lambda tup: tup[0]))
        counts = collections.Counter(map(tuple, sorted_edges))
        return sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        
    @property
    def trees(self):
        return self._mcmc_trace
        
    @property
    def trees_ll(self):
        return self._ll_trail
        
    """

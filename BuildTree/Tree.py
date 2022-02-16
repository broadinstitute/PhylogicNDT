import logging
import numpy as np
import itertools
import functools

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from Node import Node
else:
    from .Node import Node


CLONAL_CLUSTER = 1


class Tree:

    def __init__(self, nodes=None, edges=None, root=None):
        # dictionary of node id: node instance
        self._nodes = {}
        if nodes:
            self.add_nodes(nodes)
        # list of tuples (parent_id, child_id)
        self._edges = []
        if edges:
            self.add_edges(edges)
        # pointer to the root of the tree
        self._root = None
        if root:
            self.set_root(root)

    def init_tree_from_clustering(self, clustering_results):
        """ Initialize Tree object with clustering results """
        for cluster_id, cluster in clustering_results.items():
            if not cluster.blacklisted:
                self.add_node(cluster.identifier, data=cluster)
        # TODO find clonal cluster and set it as a root
        root = self.nodes[CLONAL_CLUSTER]
        self.set_root(root)
        # add edges (initially all edges are children of Clonal cluster)
        for identifier, node in self.nodes.items():
            if identifier != CLONAL_CLUSTER:
                self.add_edge(root, node)
        logging.debug('Tree initialized with edges {}'.format(self.edges))

    def __repr__(self):
        """
        Tree object representation
        """
        return repr(self._edges), [repr(node) for identifier, node in self._nodes.items()]

    def add_edge(self, parent, child):
        """
        Adds edge to the tree (tuple of node's identifiers).
        Creates new nodes if any missing
        """
        if parent:
            missing_nodes = [node for node in [parent, child] if node.identifier not in self._nodes]
            self.add_nodes(missing_nodes)
            self._edges.append((parent.identifier, child.identifier))
            # add child for parent node
            parent.add_child(child.identifier)
            # set parent for child node
            child.set_parent(parent)
        else:
            self._edges.append((None, child.identifier))

    def add_edges(self, edges):
        for (parent_id, child_id) in edges:
            if parent_id and child_id:
                self.add_edge(self._nodes[parent_id], self._nodes[child_id])

    def remove_edge(self, parent, child):
        """
        """
        # verify that nodes exist in the tree
        if parent.identifier in self._nodes and child.identifier in self._nodes:
            # verify that edge exists
            if (parent.identifier, child.identifier) in self._edges:
                self._edges.remove((parent.identifier, child.identifier))
                parent.remove_child(child.identifier)
                # TODO in case if add edge was called before remove edge
                # TODO need to enforce order of updating parent-child relationships
                if parent.identifier == child.parent.identifier:
                    child.set_parent(None)
                else:
                    # TODO add warnings
                    logging.warning('Child has different parent set')
            else:
                # TODO add warnings
                logging.warning('Warning this edge does not exists')
        else:
            # TODO add warnings
            logging.warning('One of the nodes do not exist in the list of nodes')

    def add_node(self, identifier, data=None, parent=None, children=None, root=False):
        node = Node(identifier, data, children, parent)
        if identifier and identifier not in self._nodes:
            self._nodes[identifier] = node
        else:
            logging.error('Node with this %s exists in the tree' % str(identifier))
        if root:
            self._root = node
        return node

    def set_root(self, node):
        self._root = node

    @property
    def root(self):
        return self._root

    @property
    def edges(self):
        return self._edges

    @property
    def nodes(self):
        return self._nodes

    def get_node_by_id(self, identifier):
        if identifier in self._nodes:
            return self._nodes[identifier]
        else:
            logging.debug('Node with id {} is not in the list of nodes'.format(identifier))
            return None

    def size(self):
        return len(self._nodes)

    def get_random_node(self):
        """
        :returns any node in the Tree except Clonal
        """
        # avoid picking clonal cluster (1)
        nodes_to_choose = [n for n in self._nodes.keys() if n != self._root.identifier]
        return self._nodes[np.random.choice(nodes_to_choose)]

    def add_nodes(self, nodes):
        """ Given a list of Node objects adds them to tree self._nodes dictionary """
        for node in nodes:
            self._nodes[node.identifier] = node

    def update_node(self, node):
        # if cluster is updated, not sure need it here
        raise NotImplementedError

    def remove_node(self, node):
        # don't think nodes should be removed from the tree
        raise NotImplementedError

    def traverse_by_branch(self, node=None):
        if node is None:
            node = self.root
        yield node
        for child in node.children:
            for c in self.traverse_by_branch(node=self.get_node_by_id(child)):
                yield c

    def traverse_by_level(self):
        next_level = [self.root]
        while next_level:
            yield next_level
            next_level = sum(([self.nodes[c] for c in node.children] for node in next_level), [])

    def compute_tree_likelihood(self, time_points):
        tree_llhood = sum(np.sum(self._calc_tree_lik_detailed(time_points), axis=1))
        logging.debug('Tree likelihood {}'.format(tree_llhood))
        return tree_llhood

    def _calc_tree_lik_detailed(self, time_points):
        node_llhood = []
        for identifier, node in self._nodes.items():
            parent = node.parent
            siblings = node.siblings
            if parent and len(siblings) == 0:
                def __get_tp_p_parent(tp):
                    cluster_tp_density = node.data.hist[tp]
                    # do not penalize == parent so pad by 10
                    cluster_parent_density = parent.data.hist[tp]
                    node_parent_diff = self.normalize_in_logspace(
                        self.diff_ccf(cluster_parent_density, cluster_tp_density), in_log_space=False)
                    # TODO positive log likelihood
                    if np.log(sum(node_parent_diff[101 - 5:]) + 1e-20) > 0.:
                        return 0.0
                    else:
                        return np.log(sum(node_parent_diff[101 - 5:]) + 1e-20)

                p_parent = np.sum([__get_tp_p_parent(tp) for tp in range(len(time_points))])
            else:
                p_parent = 0.

            if len(siblings) > 0:
                def __get_tp_p_sib(tp):
                    siblings_densities = []
                    for sibling in siblings:
                        siblings_densities.append(self._nodes[sibling].data.hist[tp])
                    convolved_distrib = functools.reduce(np.convolve, siblings_densities)
                    normed_conv_dist = self.normalize_in_logspace(convolved_distrib, in_log_space=False)
                    if parent:
                        parent_dist = self.normalize_in_logspace(parent.data.hist[tp], in_log_space=False)
                        parent_dist.resize(np.shape(normed_conv_dist))  # fill with zeros
                        diff_parent_sibling_dist = self.normalize_in_logspace(
                            self.diff_ccf(parent_dist, normed_conv_dist), in_log_space=False)
                        # don't penalize == sib, 10 bins
                        # parent - sum < 0 but not equal to 0
                        return np.log(sum(diff_parent_sibling_dist[len(normed_conv_dist) - 5:]) + 1e-20)
                    else:
                        # don't penalize == sib, 5 bins
                        return np.log(sum(normed_conv_dist[:101 + 5]) + 1e-20)

                p_sib = np.sum([__get_tp_p_sib(tp) for tp in range(len(time_points))])
            else:
                p_sib = 0.
            node_llhood.append([p_parent, p_sib])
        return node_llhood

    def move_node(self, node_to_move, parent=None, children=None):
        if parent:
            if node_to_move.parent:
                self.remove_edge(node_to_move.parent, node_to_move)
            self.add_edge(parent, node_to_move)
        if children:
            for child in children:
                prev_parent = child.parent
                self.remove_edge(prev_parent, child)
                self.add_edge(node_to_move, child)

    def remove_edges_for_node(self, node_to_move):
        """
        Removing edges that connect node_to_move to any other node in the Tree
        (its parent and children)
        :param node_to_move:
        :return:
        """
        import copy
        node_to_move_parent = node_to_move.parent
        node_to_move_children = copy.deepcopy(node_to_move.children)
        logging.debug('Node to move parent {}'.format(node_to_move.parent.identifier))
        # For every child of this node_to_move
        for child_id in node_to_move_children:
            logging.debug('Removing edge from node {} to node {}'.format(node_to_move.identifier, child_id))
            # Remove edge from node_to_move to its child
            self.remove_edge(node_to_move, self._nodes[child_id])
            logging.debug('Adding edge from node {} to node {}'.format(node_to_move_parent.identifier, child_id))
            # Add edge between parent of node_to_move and child of node_to_move
            self.add_edge(node_to_move_parent, self._nodes[child_id])
        # now remove edge from parent to the node_to_move
        logging.debug(
            'Removing edge from node {} to node {}'.format(node_to_move.parent.identifier, node_to_move.identifier))
        self.remove_edge(node_to_move_parent, node_to_move)

    @staticmethod
    def get_possible_configurations(potential_children):
        """ Find all possible children configurations with the list of possible children """
        new_children = []
        # Can only "borrow" up to 4 children of it's potential parent
        for n_nodes in range(1, min(len(potential_children) + 1, 4)):
            configuration = list(itertools.combinations(potential_children, n_nodes))
            new_children.extend(configuration)
        logging.debug('Possible configurations {}'.format(new_children))
        return new_children

    @staticmethod
    def logSumExp(ns):
        max_ = np.max(ns)
        ds = ns - max_
        sumOfExp = np.exp(ds).sum()
        return max_ + np.log(sumOfExp)

    def normalize_in_logspace(self, dist, in_log_space=True):
        if in_log_space:
            log_dist = np.array(dist, dtype=np.float64)
        else:
            log_dist = np.log(dist, dtype=np.float64)
        return np.exp(log_dist - self.logSumExp(log_dist))

    @staticmethod
    def diff_ccf(ccf1, ccf2):
        # Histogram of CCF1-CCF2
        ccf_dist1 = np.append(ccf1, [0] * len(ccf1))
        ccf_dist2 = np.append(ccf2, [0] * len(ccf1))
        convoluted_dist = []
        for k in range(len(ccf1)):
            inner_product = np.inner(ccf_dist1[0:len(ccf1)], ccf_dist2[len(ccf1) - 1 - k:2 * len(ccf1) - 1 - k])
            convoluted_dist.append(inner_product)
        for k in range(1, len(ccf1)):
            inner_product = np.inner(ccf_dist2[0:len(ccf1)], ccf_dist1[k:len(ccf1) + k])
            convoluted_dist.append(inner_product)
        return np.array(convoluted_dist)

    def get_all_possible_moves(self, node_to_move, time_points):
        tree_choices = []
        tree_choice_lik = []
        # Store starting tree of this iteration
        tree_choices.append(list(self.edges))
        # Store it's likelihood
        tree_choice_lik.append(self.compute_tree_likelihood(time_points))
        # Remove node from it's position the Tree
        self.remove_edges_for_node(node_to_move)
        # Get list of all potential parents (except the root and itself)
        potential_parents = [(node_id, node) for node_id, node in self._nodes.items()
                             if node_id != node_to_move.identifier]
        # Iterate over all potential parents (pp), except root
        for (pp_id, pp_node) in potential_parents:
            logging.debug('Potential parent identifier {}'.format(pp_id))
            self.move_node(node_to_move, parent=pp_node)
            # Save edges of this new Tree
            tree_choices.append(list(self.edges))
            # Compute likelihood of this new Tree
            tree_choice_lik.append(self.compute_tree_likelihood(time_points))
            # Consider all possible combinations of children
            new_children = self.get_possible_configurations(node_to_move.siblings)
            for children_configuration in new_children:
                logging.debug('Considering adding children {}'.format(children_configuration))
                # Move children to become children of the node_to_move
                for child in children_configuration:
                    self.move_node(self._nodes[child], parent=node_to_move)
                # Save edges of this new Tree
                logging.debug('New configuration edges {}'.format(self.edges))
                tree_choices.append(list(self.edges))
                # Compute likelihood of this new Tree
                tree_choice_lik.append(self.compute_tree_likelihood(time_points))
                # Reverse the move to get back to the starting Tree
                for child in children_configuration:
                    self.move_node(self._nodes[child], parent=pp_node)
                logging.debug('Restored to original edges: {}'.format(self.edges))
            self.remove_edge(pp_node, node_to_move)
        tree_choice_lik = self.normalize_in_logspace(tree_choice_lik)
        # logging.debug('Tree choice likelihoods: {}'.format(tree_choice_lik))
        return tree_choices, tree_choice_lik

    def set_new_edges(self, new_edges):
        self._edges = []
        # Clear out children's lists, children will be added in add_edge function
        for node_id, node in self._nodes.items():
            node.remove_all_children()
        # Clear out pointer to parent for each Node
        for node_id, node in self._nodes.items():
            node.remove_parent()
        # Add new edges
        for (parent_id, child_id) in new_edges:
            if child_id not in self._nodes:
                root = child_id == 1
                self.add_node(child_id, root=root)
            if parent_id:
                self.add_edge(self._nodes[parent_id], self._nodes[child_id])
            else:
                self.add_edge(None, self._nodes[child_id])

    def get_ancestry(self, node_id):
        node = self._nodes[node_id]
        current_parent = node.parent
        while current_parent:
            yield current_parent.identifier
            current_parent = current_parent.parent

    def get_tree_levels(self):
        level = 0
        children = self._root.children
        tree_levels = {}
        tree_levels[level] = [self._root.identifier]
        while len(children) > 0:
            level += 1
            tree_levels[level] = children
            curr_children = list()
            for child_id in children:
                curr_children += self._nodes[child_id].children
            children = curr_children
        logging.debug('Tree levels \n{}'.format(str(tree_levels)))
        return tree_levels

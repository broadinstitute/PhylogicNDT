import logging
import numpy as np
import itertools
import pkgutil

# Logsumexp options
from scipy.misc import logsumexp as logsumexp_scipy  # always import as double safe method (slow)

if pkgutil.find_loader('sselogsumexp') is not None:
    logging.info("Using fast logsumexp")
    from sselogsumexp import logsumexp

else:
    logging.info("Using scipy (slower) logsumexp")
    from scipy.misc import logsumexp

from Node import Node

logger = logging.getLogger(__name__)

CLONAL_CLUSTER = 1


class Tree:

    def __init__(self, nodes=None, edges=None, root=None):
        # dictionary of node id: node instance
        self._nodes = nodes if nodes else {}
        # list of tuples (parent_id, child_id)
        self._edges = edges if edges else []
        # pointer to the root of the tree
        self._root = root

    def __repr__(self):
        '''
        Represent tree
        Returns:

        '''
        return repr(self._edges), [repr(node) for identifier, node in self._nodes.items()]

    def add_edge(self, parent, child):
        '''
        Args:
            parent:
            child:
        Returns:
        '''
        missing_nodes = [node for node in [parent, child] if node.identifier not in self._nodes]

        self.add_nodes(missing_nodes)
        self._edges.append((parent.identifier, child.identifier))
        # add child for parent node
        parent.add_child(child.identifier)
        # set parent for child node
        child.set_parent(parent)

    def add_edges(self, edges):
        for (parent_id, child_id) in edges:
            self.add_edge(self._nodes[parent_id], self._nodes[child_id])

    def remove_edge(self, parent, child):
        '''

        Args:
            parent:
            child:

        Returns:

        '''
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
                    logger.warning('Child has different parent set')
            else:
                # TODO add warnings
                logger.warning('Warning this edge does not exists')
        else:
            # TODO add warnings
            logger.warning('One of the nodes do not exist in the list of nodes')

    def add_node(self, identifier, data=None, parent=None, children=None, root=False):
        node = Node(identifier, data, children, parent)
        if identifier not in self._nodes:
            self._nodes[identifier] = node
        else:
            logger.error('Node with this %s exists in the tree' % str(identifier))
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
        return self._nodes[identifier]

    def size(self):
        return len(self._nodes)

    def get_random_node(self):
        '''
        Returns:
        '''
        # avoid picking clonal cluster (1)
        nodes_to_choose = [n for n in self._nodes.keys() if n != self._root.identifier]
        return self._nodes[np.random.choice(nodes_to_choose)]

    def add_nodes(self, nodes):
        for node in nodes:
            self._nodes[node.identifier] = node

    def update_node(self, node):
        # if cluster is updated, not sure need it here
        raise NotImplementedError

    def remove_node(self, node):
        # don't think nodes should be removed from the tree
        raise NotImplementedError

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
        logger.debug('Tree levels \n{}'.format(str(tree_levels)))
        return tree_levels

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
                    node_tp_density = node.data[tp]
                    # do not penalize == parent so pad by 10
                    node_parent_diff = self.normalize_in_logspace(self.diff_ccf(parent.data[tp], node_tp_density),
                                                                  in_log_space=False)
                    # TODO positive log likelihood
                    if np.log(sum(node_parent_diff[101 - 5:]) + 1e-20) > 0.:
                        return 0.0
                    else:
                        return np.log(sum(node_parent_diff[101 - 5:]) + 1e-20)

                p_parent = np.sum([__get_tp_p_parent(tp) for tp in time_points])
            else:
                p_parent = 0.

            if len(siblings) > 0:
                def __get_tp_p_sib(tp):
                    convolved_distrib = reduce(np.convolve, [self._nodes[sibling].data[tp] for sibling in siblings])
                    normed_conv_dist = self.normalize_in_logspace(convolved_distrib, in_log_space=False)
                    if parent:
                        parent_dist = self.normalize_in_logspace(parent.data[tp], in_log_space=False)
                        parent_dist.resize(np.shape(normed_conv_dist))  # fill with zeros
                        diff_parent_sibling_dist = self.normalize_in_logspace(
                            self.diff_ccf(parent_dist, normed_conv_dist), in_log_space=False)
                        # don't penalize == sib, 10 bins
                        # parent - sum < 0 but not equal to 0
                        return np.log(sum(diff_parent_sibling_dist[len(normed_conv_dist) - 5:]) + 1e-20)
                    else:
                        # don't penalize == sib, 5 bins
                        return np.log(sum(normed_conv_dist[:101 + 5]) + 1e-20)

                p_sib = np.sum([__get_tp_p_sib(tp) for tp in time_points])
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
        '''
        Removing edges that connect node_to_move to any other node in the Tree
        (its parent and children)
        :param node_to_move:
        :return:
        '''
        node_to_move_parent = node_to_move.parent
        logging.debug('Node to move parent {}'.format(node_to_move.parent.identifier))
        # For every child of this node_to_move
        for child_id in node_to_move.children:
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
        """

        :param potential_children:
        :return:
        """
        new_children = []
        # Can only "borrow" up to 3 children of it's potential parent
        for n_nodes in range(1, min(len(potential_children) + 1, 4)):
            configuration = list(itertools.combinations(potential_children, n_nodes))
            new_children.extend(configuration)
        logging.debug('Possible configurations {}'.format(new_children))
        return new_children

    @staticmethod
    def normalize_in_logspace(dist, in_log_space=True):
        if not in_log_space:
            log_dist = np.log(dist, dtype=np.float64)
            return np.exp(log_dist - logsumexp_scipy(log_dist))
        else:
            logging.debug('Likelihood before normalization\n{}'.format(dist))
            log_dist = np.array(dist, dtype=np.float64)
            return np.exp(log_dist - logsumexp_scipy(log_dist))

    """
    @staticmethod
    def diff_ccf(ccf1, ccf2, difference=True):
        # reduce(np.convolve, [self.clusters[x][s_idx] for x in siblings])
        if difference:
            return np.convolve(ccf1, ccf2[::-1])
        else:
            return np.convolve(ccf1, ccf2)
    """

    def diff_ccf(self, ccf1, ccf2, difference=True):
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
                tree_choices.append(list(self.edges))
                # Compute likelihood of this new Tree
                tree_choice_lik.append(self.compute_tree_likelihood(time_points))
                # Reverse the move to get back to the starting Tree
                for child in children_configuration:
                    self.move_node(self._nodes[child], parent=node_to_move)
            self.remove_edge(pp_node, node_to_move)
        tree_choice_lik = self.normalize_in_logspace(tree_choice_lik)
        logging.debug('Tree choice likelihoods \n {}'.format(tree_choice_lik))
        return tree_choices, tree_choice_lik

    def set_new_edges(self, new_edges):
        self._edges = []
        # Clear out children's lists, children will be added in add_edge function
        for node_id, node in self._nodes.items():
            node.remove_all_children()
        for node_id, node in self._nodes.items():
            node.remove_parent()
        # Create new edges
        for (parent_id, child_id) in new_edges:
            self.add_edge(self._nodes[parent_id], self._nodes[child_id])

    def get_ancestry(self, node_id):
        node = self._nodes[node_id]
        ancestry = [node_id]
        current_parent = node.parent
        while current_parent:
            ancestry.append(current_parent.identifier)
            current_parent = current_parent.parent
        logging.debug('Ancestory for node {} is {}'.format(node_id, ancestry[::-1]))
        return ancestry[::-1]

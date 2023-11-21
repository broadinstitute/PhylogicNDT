import logging


class Node:

    def __init__(self, identifier, data, children=None, parent=None):
        self._identifier = identifier
        # ClusterObject
        self._data = data
        self._parent = parent if parent else None
        if children is None:
            self._children = []
        else:
            self._children = children

    def __eq__(self, other):
        """ """
        if other is None:
            return False
        if self._data == other.data:  # TODO decide if the same identifiers are also required for equality
            return True
        else:
            return False

    def __hash__(self):
        return hash(str(self._identifier))

    def __repr__(self):
        if self._data:
            return self._identifier, [(sample_id + str(density)) for sample_id, density in self._data.items()]
        else:
            return str(self._identifier) + ''

    def add_child(self, child_id):
        """ """
        self._children.append(child_id)

    def add_children(self, new_children):
        for child_id in new_children:
            self.add_child(child_id)

    def set_parent(self, parent):
        """ """
        self._parent = parent

    def remove_child(self, child_id):
        """ """
        if child_id in self._children:
            self._children.remove(child_id)
        else:
            logging.error('Node %s does not have child with id %s' % (str(child_id), str(self._identifier)))

    def remove_all_children(self):
        self._children = []

    def remove_parent(self):
        self._parent = None

    @property
    def identifier(self):
        """ """
        return self._identifier

    @property
    def data(self):
        """ """
        return self._data

    @property
    def children(self):
        """ """
        return self._children

    @property
    def parent(self):
        """ """
        return self._parent

    @property
    def siblings(self):
        """ """
        if self.parent:
            siblings = self.parent.children
            return [node_id for node_id in siblings if node_id != self.identifier]
        else:
            return []

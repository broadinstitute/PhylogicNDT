class Cluster:

    def __init__(self, identifier, densities=None, mutations=None):
        self._identifier = identifier
        self._densities = densities if densities else {}
        self._mutations = mutations if mutations else []

    def __eq__(self, other):
        '''

        Args:
            other:

        Returns:

        '''
        if other is None:
            return False
        # TODO decide if the same identifiers are also required for equality
        if self._mutations == other.mutations and self._ccf == other.ccf:
            return True
        else:
            return False

    @property
    def identifier(self):
        '''

        Returns:

        '''
        return self._identifier

    @property
    def densities(self):
        '''

        Returns:

        '''
        return self._densities

    @property
    def mutations(self):
        '''

        :return:
        '''
        return self._mutations

    def add_mutation(self, mutation):
        '''

        Args:
            mutation:

        Returns:

        '''
        self._mutations.append(mutation)

    def remove_mutation(self, mutation):
        raise NotImplementedError

    def _update_densities(self):
        raise NotImplementedError

    def add_mutations(self, mutations):
        for mutation in mutations:
            self.add_mutation(mutation)

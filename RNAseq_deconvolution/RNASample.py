'''
Sample class to load and store RNAseq expression data for each sample of the individual
'''

import os
import logging

na_values = {'-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A', 'N/A', 'NA', '#NA', 'NULL', 'NaN', '-NaN', 'nan',
             '-nan', ''}  # http://pandas.pydata.org/pandas-docs/stable/io.html#na-values


# TODO: Need a way to flag genes that have TPM=0 across all timepoints
# TODO: Hugo symbols for genes? Store it as a separate mapping {gene_id -> gene_name}


class RNASample:
    """ """

    common_field_map = {"gene_id", "transcript_id(s)", "length", "TPM"}

    def __init__(self, file_name,
                 input_type,
                 indiv=None,
                 sample_name=None,
                 sample_short_id=None,
                 DNAsource=None,
                 timepoint=None,
                 # By default loads file with Mitochondrial genes
                 gene_blacklist=os.path.join(os.path.dirname(__file__), 'supplement/Blacklist_Genes.txt'),
                 purity=None):

        # Reference to Patient object
        self._indiv = indiv
        self._inut_type = input_type
        self._sample_name = sample_name
        self._sample_short_id = sample_short_id
        self._DNAsource = DNAsource
        self._timepoint = timepoint
        self._purity = purity
        # Blacklist genes from the analysis (Gene id and name is from gencode_v19)
        self._gene_blacklist = self._load_blacklist_from_file(gene_blacklist)
        # a dictionary hash table for fast TPM value lookup
        self._tpm_values = self._load_tpm_values(file_name)

    @staticmethod
    def _load_blacklist_from_file(in_file):
        gene_blacklist = {}
        with open(in_file, 'r') as reader:
            for line in reader:
                values = line.strip().split('\t')
                gene_blacklist[values[0]] = values[1]
        return gene_blacklist

    def _load_tpm_values(self, in_file):
        tpm_values_dict = {}
        with open(in_file) as reader:
            for line in reader:
                values = line.strip('\n').split('\t')
                if line.startswith('gene_id'):
                    header = {x: i for i, x in enumerate(values)}
                else:
                    gene_id = values[header['gene_id']]
                    if gene_id not in self._gene_blacklist:
                        try:
                            tpm_value = float(values[header['TPM']])
                        except ValueError:
                            tpm_value = 0
                        tpm_values_dict[gene_id] = tpm_value
                    else:
                        logging.debug('Gene {} is in blacklist'.format(gene_id))
        return tpm_values_dict

    @property
    def purity(self):
        return self._purity

    @property
    def sample_name(self):
        return self._sample_name

    @property
    def timepoint(self):
        return self._timepoint

    def get_tpm_by_gene_id(self, gene_id):
        return self._tpm_values[gene_id]

    def get_tpm_by_gene_name(self, gene_name):
        # Handle cases where gene names that are duplicate
        pass

    @property
    def tpm_values(self):
        return self._tpm_values

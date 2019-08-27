import logging
import numpy as np

from scipy.special import logsumexp as logsumexp_scipy

import data.SomaticEvents as SomaticEvents
from .ClusterObject import Cluster


class ClusteringResults:

    def __init__(self, mut_info_file, cluster_info_file):
        self._cluster_mutations = {}
        self._samples_mutations = {}
        self._clusters = {}
        self._removed_clusters = []
        self._create_clusters(cluster_info_file)
        self._load_mutations(mut_info_file)

    def _load_mutations(self, mut_info_file):
        logging.debug('Loading mutations from {} file'.format(mut_info_file))
        ccf_headers = ['preDP_ccf_' + str(i / 100.0) for i in range(0, 101, 1)]
        with open(mut_info_file, 'r') as reader:
            for line in reader:
                values = line.strip().split('\t')
                if line.startswith('Patient_ID'):
                    header = dict((item, idx) for idx, item in enumerate(values))
                elif values[header['Variant_Type']] != 'CNV':
                    # TODO: for reshuffling need to keep all mutations and clusters
                    cluster_id = int(values[header['Cluster_Assignment']])
                    if cluster_id not in self._removed_clusters:
                        chromosome = values[header['Chromosome']]
                        position = values[header['Start_position']]
                        ref = values[header['Reference_Allele']]
                        alt = values[header['Tumor_Seq_Allele']]
                        sample_id = values[header['Sample_ID']]
                        ccf_1d = [float(values[header[i]]) for i in ccf_headers]
                        ccf_1d = np.clip(np.array(ccf_1d, dtype=np.float64), a_min=1e-20, a_max=None)
                        ccf_1d = np.log(ccf_1d, dtype=np.float64)
                        ccf_1d = np.exp(ccf_1d - logsumexp_scipy(ccf_1d))
                        var_type = values[header['Variant_Type']]
                        mutation_str = ':'.join([chromosome, position, ref, alt])
                        if cluster_id not in self._cluster_mutations:
                            self._cluster_mutations[cluster_id] = {}
                        if mutation_str not in self._cluster_mutations[cluster_id]:
                            self._cluster_mutations[cluster_id][mutation_str] = {}

                        if sample_id not in self._samples_mutations:
                            self._samples_mutations[sample_id] = []
                        t_ref_count = self._get_count(values[header['t_ref_count']])
                        t_alt_count = self._get_count(values[header['t_alt_count']])

                        mutation = SomaticEvents.SomMutation(chromosome, position, ref, alt, ccf_1d,
                                                             ref_cnt=t_ref_count,
                                                             alt_cnt=t_alt_count,
                                                             gene=values[header['Hugo_Symbol']],
                                                             prot_change=values[header['Protein_change']],
                                                             mut_category=values[header['Variant_Classification']],
                                                             from_sample=sample_id,
                                                             type_=var_type)

                        self._cluster_mutations[cluster_id][mutation_str][sample_id] = mutation
                        self._samples_mutations[sample_id].append(mutation_str)
                        self._clusters[cluster_id].add_mutation(mutation)
                        logging.info('Mutation {} loaded from sample {}'.format(mutation_str, sample_id))

    @staticmethod
    def _get_count(count):
        try:
            return float(count)
        except ValueError:
            return None

    def _load_clusters(self, cluster_info_file):
        logging.debug('Loading clusters from {} file'.format(cluster_info_file))
        cluster_ccf = {}
        means = {}
        ccf_headers = ['postDP_ccf_' + str(i / 100.0) for i in range(0, 101, 1)]
        with open(cluster_info_file, 'r') as reader:
            for line in reader:
                values = line.strip().split('\t')
                if line.startswith('Patient_ID'):
                    header = dict((item, idx) for idx, item in enumerate(values))
                else:
                    sample_id = values[header['Sample_ID']]
                    cluster_id = int(values[header['Cluster_ID']])
                    cluster_mean = float(values[header['postDP_ccf_mean']])
                    ccf = np.array([float(values[header[i]]) for i in ccf_headers], dtype=np.float64)
                    ccf = np.clip(ccf, a_min=1e-20, a_max=None)
                    ccf = np.log(ccf, dtype=np.float64)
                    ccf = np.exp(ccf - logsumexp_scipy(ccf))
                    if cluster_id not in cluster_ccf:
                        cluster_ccf[cluster_id] = {}
                        means[cluster_id] = []
                    means[cluster_id].append(cluster_mean)
                    cluster_ccf[cluster_id][sample_id] = ccf
        for cluster_id in cluster_ccf:
            # decide whether cluster should be removed
            # if density < 0.1 across all samples add it to remove clusters, to be removed from BuildTree algorithm
            if self.low_ccf_check(means[cluster_id]):
                self._removed_clusters.append(cluster_id)
                logging.debug('Removed cluster {} '.format(cluster_id))
        return cluster_ccf

    @staticmethod
    def low_ccf_check(cluster_means):
        return all([ccf_mean < 0.1 for ccf_mean in cluster_means])

    def _create_clusters(self, cluster_info_file):
        clusters_ccf = self._load_clusters(cluster_info_file)
        for cluster_id, densities in clusters_ccf.items():
            if cluster_id not in self._removed_clusters:
                self._clusters[cluster_id] = Cluster(cluster_id, densities=densities)
                logging.debug('Created cluster {} '.format(cluster_id))

    @property
    def clusters(self):
        return self._clusters

    @property
    def samples(self):
        return list(self._samples_mutations.keys())

    @property
    def time_points(self):
        return len(self._samples_mutations)

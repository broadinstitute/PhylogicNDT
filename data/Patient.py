##########################################################
# Patient - central class to handle and store sample CCF data
##########################################################
from Sample import TumorSample
from Sample import RNASample
from SomaticEvents import SomMutation, CopyNumberEvent
from Enums import CSIZE, CENT_LOOKUP
import os
import sys
import logging
import itertools

import numpy as np
from intervaltree import Interval, IntervalTree

# Logsumexp options
import pkgutil

if pkgutil.find_loader('sselogsumexp') is not None:
    from sselogsumexp import logsumexp
else:
    from scipy.misc import logsumexp  # LogAdd


class Patient:
    """CLASS info

     FUNCTIONS:
         public:

           add_sample() - add new samples from individual to self.sample_list by making a new TumorSample instance

        private:

           _auto_file_type() - guess input file type for ccf data if not manually provided

     PROPERTIES:
         regular variables:
            self.sample_list

         @property methods: none
    """

    def __init__(self, indiv_name='Indiv1',
                 ccf_grid_size=101,
                 driver_genes_file=os.path.join(os.path.dirname(__file__), 'supplement_data/Driver_genes_v1.0.txt'),
                 impute_missing=False,
                 artifact_blacklist=os.path.join(os.path.dirname(__file__), 'supplement_data/Blacklist_SNVs.txt'),
                 artifact_whitelist='',
                 use_indels=False,
                 min_coverage=8,
                 PoN_file=False):

        # DECLARATIONS
        self.indiv_name = indiv_name
        # :type : list [TumorSample]
        self.sample_list = []
        self.rna_sample_list = []

        self.samples_synchronized = False
        self.rna_samples_synchronized = False
        self.concordant_genes = []
        self.driver_genes = self._parse_driver_g_file(driver_genes_file)
        self.ccf_grid_size = ccf_grid_size

        self.PatientLevel_MutBlacklist = artifact_blacklist
        self.PatientLevel_MutWhitelist = artifact_whitelist

        # Patient configuration settings
        # flag if to impute missing variants as ccf 0
        self.impute_missing = impute_missing
        # min cov is specified both here and passed to tumor sample.
        self.min_coverage = min_coverage
        self.use_indels = use_indels
        self.PoN_file = PoN_file

        self._validate_sample_names()

        # later filled data objects
        self.ND_mutations = []

        # storing of results
        # Clustering
        self.ClusteringResults = None
        self.MutClusters = None
        self.TruncalMutEvents = None
        self.MCMC_trace = None
        self.k_trace = None
        self.alpha_trace = None

        self.unclustered_muts = []

        self.concordant_cn_tree = {chrom: IntervalTree() for chrom in list(map(str, range(1, 23))) + ['X', 'Y']}

        # BuildTree
        self.TopTree = None
        self.TreeEnsemble = []
        self.alt_cn_states = []

    def initPatient(self):
        """ accepted input types abs; txt; sqlite3 .db # auto tab if .txt, .tsv or .tab ; abs if .Rdata; sqlite if .db """
        raise NotImplementedError

    def addRNAsample(self, filen, sample_name, input_type='auto', purity=None, timepoint=None):
        """ Accepted input types rsem.genes """
        # make new sample and add to exiting list of samples
        logging.info("Adding expression from RNA Sample: %s", sample_name)
        new_rna_sample = RNASample(filen, input_type, indiv=self.indiv_name, sample_name=sample_name,
                                   timepoint=timepoint, purity=purity)
        self.rna_sample_list.append(new_rna_sample)
        logging.info('Added RNA sample ' + new_rna_sample.sample_name)
        # turn of concordance flag when new sample is added
        self.rna_samples_synchronized = False

    def preprocess_rna_samples(self):

        count_low_exp_genes = 0
        gene_ids = self.rna_sample_list[0].get_gene_ids()
        for gene_id in gene_ids:
            tpm_values = []
            for sample in self.rna_sample_list:
                tpm_values.append(sample.get_tpm_by_gene_id(gene_id))
            if all(i <= 1.0 for i in tpm_values):
                count_low_exp_genes += 1
            else:
                self.concordant_genes.append(gene_id)
        self.rna_samples_synchronized = True
        logging.debug('{} genes have TPM values less a than 1.0 across all timepoints.'.format(count_low_exp_genes))
        logging.debug('{} genes will be used in the analysis.'.format(len(self.concordant_genes)))

    def addSample(self, filen, sample_name, input_type='auto', seg_input_type='auto', grid_size=101, seg_file=None,
                  _additional_muts=None, purity=None, timepoint_value=None):
        """ accepted input types abs; txt; sqlite3 .db # auto tab if .txt, .tsv or .tab ; abs if .Rdata; sqlite if .db"""

        if _additional_muts == []:
            _additional_muts = None
        elif type(_additional_muts) is list:
            _additional_muts = _additional_muts[0]

        # make new sample and add to exiting list of samples
        logging.info("Adding Mutations from Sample: %s", sample_name)
        new_sample = TumorSample(filen, input_type, seg_input_type=seg_input_type, sample_name=sample_name,
                                 artifact_blacklist=self.PatientLevel_MutBlacklist,
                                 artifact_whitelist=self.PatientLevel_MutWhitelist,
                                 ccf_grid_size=grid_size, PoN=self.PoN_file, indiv=self.indiv_name,
                                 use_indels=self.use_indels, min_coverage=self.min_coverage,
                                 _additional_muts=_additional_muts, seg_file=seg_file,
                                 purity=purity, timepoint_value=timepoint_value)

        self.sample_list.append(new_sample)
        logging.info('Added sample ' + new_sample.sample_name)
        # turn of concordance flag when new sample is added
        self.samples_synchronized = False

    def homogenize_events_across_samples(self):
        # TODO: do not understand this function, find place where it is used
        if len(self.sample_list) == 1:
            UserWarning("Found one sample! Will run 1D clustering. If intended to run ND, please fix!")

    def get_sample_byname(self, sample_name):
        """
        :param str sample_name: sample name to search for
        :return: TumorSample
        """
        for smpl in self.sample_list:
            if smpl.sample_name == sample_name:
                return smpl
        logging.warning("Sample with the name {} is not found in the sample list".format(sample_name))
        return None

    def _validate_sample_names(self, disallowed_strings=['Inner']):
        sample_names = self.sample_names
        # check that no samples have the same name
        if len(sample_names) != len(set(sample_names)):
            logging.error('Several samples appear to have identical names!')
            ValueError('Several samples appear to have identical names! This is not allowed! Please fix!')
        # check that no samples disallowed strings in its name
        for dis_str in disallowed_strings:
            for smpl_name in sample_names:
                if dis_str in smpl_name:
                    logging.error('Disallowed string found in sample name {}'.format(dis_str))
                    ValueError('Several samples appear to have identical names! This is not allowed! Please fix!')

        return True

    @property
    def sample_names(self):
        return [x.sample_name for x in self.sample_list]

    @staticmethod
    def _parse_driver_g_file(filen):
        """ read driver file as one gene per line """
        if not filen:
            return set()  # empty list
        with open(filen, 'r') as drv_file:
            drv = [x.strip() for x in drv_file.read().strip().split('\n')]
        return set(drv)

    def preprocess_samples(self):
        """ Central preprocessing of samples
            Makes sure that there is info on all mutations and everything is concordant on all samples """
        if len(self.sample_list) < 2:
            logging.warning("Only one sample in sample set! Cannot check concordance for multi-D clustering!")
            logging.warning("Cleaning this single sample!")

            # TODO: port the 1D-preprocess
            for sample in self.sample_list:  # because of imputing: has to be done at the end
                sample.concordant_variants = sample.mutations
                # sort concordant variants by presence in driver then var_str/gene name
                sample.concordant_variants.sort(
                    key=lambda x: (str(x).split('_')[0] in self.driver_genes, str(x), x.var_str))
                sample.private_mutations = [mut for mut in sample.concordant_variants if mut.alt_cnt > 0]

        self._validate_sample_names()
        try:
            blacklist = set(set.union(*[set(x.known_blacklisted_mut) for x in self.sample_list]))
        except TypeError:
            logging.info("Blacklist is not specified.")
            blacklist = set()

        # use var stings from sample1 to order all other samples
        var_str_init = self.sample_list[0].mut_varstr
        count_needed = len(self.sample_list)
        full_var_list = list(itertools.chain.from_iterable([x.mut_varstr for x in self.sample_list]))
        vars_present_in_all = set()

        for mut in var_str_init:
            if full_var_list.count(mut) == count_needed and mut not in blacklist:
                vars_present_in_all.add(mut)

        joint_temporarily_removed = set()
        for sample in self.sample_list:
            joint_temporarily_removed.update(sample.temporarily_removed)

        joint_temporarily_removed = set([mut.var_str for mut in joint_temporarily_removed])
        # Only allow one fixed CNV in concordant variants.
        first_cnv = None
        # iterate through all samples and make concordance sets, then sort them
        for sample in self.sample_list:
            if not self.impute_missing:
                # reset any previous joint results ; if imputing leave as is
                sample.concordant_variants = []
            sample.concordant_with_samples = []

            for mut in sample.mutations + sample.low_coverage_mutations.values():

                # TODO: Add this as a flag
                # if mut.mut_category is not None and ("rna" in mut.mut_category.lower() or "utr" in mut.mut_category.lower()):
                #    logging.warning("REMOVING RNA MUTATION:" + str(mut))
                #    sample.artifacts_in_blacklist.append(mut)
                #    blacklist.add(mut.var_str)
                #

                if mut.var_str in blacklist:  # If blacklisted in another sample, remove mutation.
                    logging.info("Found blacklisted mutation still in sample, removing:" + str(mut))
                    if mut.var_str not in sample.known_blacklisted_mut:
                        sample.known_blacklisted_mut.add(mut.var_str)
                    continue

                elif mut.var_str in vars_present_in_all:
                    if mut.var_str not in joint_temporarily_removed:
                        if mut.type == "CNV":
                            if first_cnv is None:
                                first_cnv = mut.var_str
                        if mut.type != "CNV" or first_cnv == mut.var_str:
                            sample.concordant_variants.append(mut)
                    else:
                        continue  # nothing to do for indels in all samples

                elif mut.var_str not in vars_present_in_all:
                    if mut.var_str in joint_temporarily_removed:
                        sample.artifacts_in_blacklist.append(mut)
                        logging.warn(
                            "Forcecalling Failure? Mutation {} not in {} but otherwise graylisted already.".format(
                                str(mut), sample.sample_name))
                        continue

                    elif mut.var_str not in joint_temporarily_removed and self.impute_missing:
                        for mis_sample in [x for x in self.sample_list if x != sample]:
                            if mut not in mis_sample.concordant_variants and mut not in mis_sample.mutations and mut.var_str not in joint_temporarily_removed and ":".join(
                                    map(str,
                                        [mut.chrN, mut.pos, mut.ref, mut.alt])) not in mis_sample.known_blacklisted_mut:
                                logging.info('Imputing missing mutation as 0 CCF, sample %s, var: %s ; %s',
                                             sample.sample_name, mut, mut.var_str)
                                mis_sample.concordant_variants.append(
                                    SomMutation.from_som_mutation_zero(mut, from_sample=mis_sample))
                                mis_sample._mut_varstring_hashtable[mut.var_str] = mis_sample.concordant_variants[-1]
                        sample.concordant_variants.append(mut)

                    else:
                        logging.error(
                            'Mutation missing in some datasets, sample %s, var: %s . This mutation will be skipped. Use --impute, or forcecall the mutations.',
                            sample.sample_name, mut)

        for sample in self.sample_list:  # because of imputing: has to be done at the end
            # sort concordant variants by presence in driver then var_str/gene name
            sample.concordant_variants.sort(
                key=lambda x: (str(x).split('_')[0] in self.driver_genes, str(x), x.var_str))
            # annotate each sample with what samples were used for concordance
            sample.concordant_with_samples = self.sample_list
        # turn on concordance flag when new sample mut are joined
        self.samples_synchronized = True

    def _make_ND_histogram(self):
        if not self.samples_synchronized:
            logging.error("Could not make ND Histogram, please make sure to call preprocess_samples()")
            return False

        combined_ccf = []
        for mut in self.sample_list[0].concordant_variants:
            mut_nd_ccf = [mut.ccf_1d]
            for sample in self.sample_list[1:]:
                mut_nd_ccf.append(sample.get_mut_by_varstr(mut.var_str).ccf_1d)
            combined_ccf.append(mut_nd_ccf)

        return NDHistogram(combined_ccf, [x.var_str for x in self.sample_list[0].concordant_variants])

    def make_ND_histogram(self):
        if not self.samples_synchronized:
            logging.error("Could not make ND Histogram, please make sure to call preprocess_samples()")
            return False
        combined_ccf = []
        for mut in self.sample_list[0].concordant_variants:
            mut_nd_ccf = [mut.ccf_1d]
            for sample in self.sample_list[1:]:
                mut_nd_ccf.append(sample.get_mut_by_varstr(mut.var_str).ccf_1d)
            combined_ccf.append(mut_nd_ccf)

        return NDHistogram(combined_ccf, [x.var_str for x in self.sample_list[0].concordant_variants])

    def cluster_temp_removed(self):
        clust_CCF_results = self.ClusteringResults.clust_CCF_dens
        for mut in self.sample_list[0].low_coverage_mutations.values():
            mut_coincidence = np.ones(len(clust_CCF_results))
            for i, sample in enumerate(self.sample_list):
                try:
                    mut = sample.get_mut_by_varstr(mut.var_str)
                except KeyError:
                    logging.warning(mut.var_str + ' not called across all samples')
                    mut_coincidence.fill(np.nan)
                    break
                for ii, cluster_ccfs in enumerate(clust_CCF_results):
                    cluster_ccf = cluster_ccfs[i]
                    if abs(np.argmax(mut.ccf_1d) - np.argmax(cluster_ccf)) > 70:
                        dot = 0.
                    else:
                        dot = max(sum(np.array(mut.ccf_1d) * cluster_ccf), .0001)
                    mut_coincidence[ii] *= dot
            if np.any(mut_coincidence > 0.):
                cluster_assignment = np.argmax(mut_coincidence) + 1
                for i, sample in enumerate(self.sample_list):
                    mut = sample.get_mut_by_varstr(mut.var_str)
                    mut.cluster_assignment = cluster_assignment
                    mut.clust_ccf = clust_CCF_results[cluster_assignment - 1][i]
            else:
                print('Did not cluster ' + str(mut))
                self.unclustered_muts.append(mut.var_str)
                for sample in self.sample_list:
                    try:
                        mut = sample.low_coverage_mutations[mut.var_str]
                        sample.unclustered_muts.append(mut)
                    except KeyError:
                        print(mut.var_str + ' not found in ' + sample.sample_name)
                # for sample in self.sample_list:
                #     mut = sample.get_mut_by_varstr(mut.var_str)
                #     mut.cluster_assignment = None
                #     mut.clust_ccf = None

    # def intersect_cn_trees(self):
    #     """
    #     Gets copy number events from segment trees and adds them to samples
    #
    #     """
    #     def get_bands(chrom, start, end, cytoband=os.path.dirname(__file__) + '/supplement_data/cytoBand.txt'):
    #         """
    #         Gets cytobands hit by a CN event
    #
    #         """
    #         bands = []
    #         on_c = False
    #         with open(cytoband, 'r') as f:
    #             for line in f:
    #                 row = line.strip('\n').split('\t')
    #                 if row[0].strip('chr') != str(chrom):
    #                     if on_c:
    #                         return bands
    #                     continue
    #                 if int(row[1]) <= end and int(row[2]) >= start:
    #                     bands.append(Cytoband(chrom, row[3]))
    #                     on_c = True
    #                 if int(row[1]) > end:
    #                     return bands
    #
    #     def merge_cn_events(event_segs, neighbors, R=frozenset(), X=frozenset()):
    #         """
    #         Merges copy number events on a single chromosome if they are adjacent and their ccf values are similar
    #
    #         Args:
    #             event_segs: set of CN segments represented as tuple(bands, CNs, CCF_hats, CCF_highs, CCF_lows, allele)
    #             neighbors: dict mapping seg to set of neighbors (segs with similar CCFs)
    #             R: only populated in recursive calls
    #             X: only populated in recursive calls
    #
    #         Returns:
    #             Generator for merged segs
    #         """
    #         is_max = True
    #         for s in itertools.chain(event_segs, X):
    #             if isadjacent(s, R):
    #                 is_max = False
    #                 break
    #         if is_max:
    #             bands = set.union(*(set(b[0]) for b in R))
    #             cns = next(iter(R))[1]
    #             ccf_hat = np.zeros(len(self.sample_list))
    #             ccf_high = np.zeros(len(self.sample_list))
    #             ccf_low = np.zeros(len(self.sample_list))
    #             for seg in R:
    #                 ccf_hat += np.array(seg[2])
    #                 ccf_high += np.array(seg[3])
    #                 ccf_low += np.array(seg[4])
    #             yield (bands, cns, ccf_hat / len(R), ccf_high / len(R), ccf_low / len(R))
    #         else:
    #             for s in event_segs:
    #                 if isadjacent(s, R):
    #                     for region in merge_cn_events(event_segs & neighbors[s], neighbors, R=R | {s}, X=X & neighbors[s]):
    #                         yield region
    #                     event_segs = event_segs - {s}
    #                     X = X | {s}
    #
    #     def isadjacent(s, R):
    #         """
    #         Copy number events are adjacent if the max band of one is the same as
    #         or adjacent to the min band of the other
    #
    #         """
    #         if not R:
    #             return True
    #         Rchain = list(itertools.chain(*(b[0] for b in R)))
    #         minR = min(Rchain)
    #         maxR = max(Rchain)
    #         mins = min(s[0])
    #         maxs = max(s[0])
    #         if mins >= maxR:
    #             return mins - maxR <= 1 and mins.band[0] == maxR.band[0]
    #         elif maxs <= minR:
    #             return minR - maxs <= 1 and maxs.band[0] == minR.band[0]
    #         else:
    #             return False
    #
    #     c_trees = {}
    #     n_samples = len(self.sample_list)
    #     for chrom in list(map(str, range(1, 23)))+['X', 'Y']:
    #         tree = IntervalTree()
    #         for sample in self.sample_list:
    #             if sample.CnProfile:
    #                 tree.update(sample.CnProfile[chrom])
    #         tree.split_overlaps()
    #         tree.merge_equals(data_initializer=[], data_reducer=lambda a, c: a + [c])
    #         c_tree = IntervalTree(filter(lambda s: len(s.data) == n_samples, tree))
    #         c_trees[chrom] = c_tree
    #         event_segs = set()
    #         for seg in c_tree:
    #             start = seg.begin
    #             end = seg.end
    #             bands = get_bands(chrom, start, end)
    #             cns_a1 = []
    #             cns_a2 = []
    #             ccf_hat_a1 = []
    #             ccf_hat_a2 = []
    #             ccf_high_a1 = []
    #             ccf_high_a2 = []
    #             ccf_low_a1 = []
    #             ccf_low_a2 = []
    #             for i, sample in enumerate(self.sample_list):
    #                 seg_data = seg.data[i][1]
    #                 cns_a1.append(seg_data['cn_a1'])
    #                 cns_a2.append(seg_data['cn_a2'])
    #                 ccf_hat_a1.append(seg_data['ccf_hat_a1'] if seg_data['cn_a1'] != 1 else 0.)
    #                 ccf_hat_a2.append(seg_data['ccf_hat_a2'] if seg_data['cn_a2'] != 1 else 0.)
    #                 ccf_high_a1.append(seg_data['ccf_high_a1'] if seg_data['cn_a1'] != 1 else 0.)
    #                 ccf_high_a2.append(seg_data['ccf_high_a2'] if seg_data['cn_a2'] != 1 else 0.)
    #                 ccf_low_a1.append(seg_data['ccf_low_a1'] if seg_data['cn_a1'] != 1 else 0.)
    #                 ccf_low_a2.append(seg_data['ccf_low_a2'] if seg_data['cn_a2'] != 1 else 0.)
    #             cns_a1 = np.array(cns_a1)
    #             cns_a2 = np.array(cns_a2)
    #             if np.all(cns_a1 == 1):
    #                 pass
    #             elif np.all(cns_a1 >= 1) or np.all(cns_a1 <= 1):
    #                 event_segs.add((tuple(bands), tuple(cns_a1), tuple(ccf_hat_a1), tuple(ccf_high_a1), tuple(ccf_low_a1), 'a1'))
    #             else:
    #                 logging.warning('Seg with inconsistent event: {}:{}:{}'.format(chrom, seg.begin, seg.end))
    #             if np.all(cns_a2 == 1):
    #                 pass
    #             elif np.all(cns_a2 >= 1) or np.all(cns_a2 <= 1):
    #                 event_segs.add((tuple(bands), tuple(cns_a2), tuple(ccf_hat_a2), tuple(ccf_high_a2), tuple(ccf_low_a2), 'a2'))
    #             else:
    #                 logging.warning('Seg with inconsistent event: {}:{}:{}'.format(chrom, seg.begin, seg.end))
    #         neighbors = {s: set() for s in event_segs}
    #         for seg1, seg2 in itertools.combinations(event_segs, 2):
    #             s1_hat = np.array(seg1[2])
    #             s2_hat = np.array(seg2[2])
    #             if seg1[1] == seg2[1] and np.all(s1_hat >= np.array(seg2[4])) and np.all(s1_hat <= np.array(seg2[3]))\
    #             and np.all(s2_hat >= np.array(seg1[4])) and np.all(s2_hat <= np.array(seg1[3])):
    #                 neighbors[seg1].add(seg2)
    #                 neighbors[seg2].add(seg1)
    #
    #         event_cache = []
    #         if event_segs:
    #             for bands, cns, ccf_hat, ccf_high, ccf_low in merge_cn_events(event_segs, neighbors):
    #                 mut_category = 'gain' if sum(cns) > len(self.sample_list) else 'loss'
    #                 a1 = (mut_category, bands) not in event_cache
    #                 if a1:
    #                     event_cache.append((mut_category, bands))
    #                 self._add_cn_event_to_samples(chrom, min(bands), max(bands), cns, mut_category, ccf_hat, ccf_high,
    #                     ccf_low, a1, dupe=not a1)
    #     self.concordant_cn_tree = c_trees

    # def get_arm_level_cn_events(self, size_threshold=.4):
    #     chromosomes = list(map(str, range(1, 23))) + ['X']
    #     for chrN, arm in itertools.product(chromosomes, 'pq'):
    #         centromere

    def get_arm_level_cn_events(self):
        n_samples = len(self.sample_list)
        for ckey, (chrom, csize) in enumerate(zip(list(map(str, range(1, 23))) + ['X', 'Y'], CSIZE)):
            centromere = CENT_LOOKUP[ckey + 1]
            tree = IntervalTree()
            for sample in self.sample_list:
                if sample.CnProfile:
                    tree.update(sample.CnProfile[chrom])
            tree.split_overlaps()
            tree.merge_equals(data_initializer=[], data_reducer=lambda a, c: a + [c])
            c_tree = IntervalTree(filter(lambda s: len(s.data) == n_samples, tree))
            event_segs = set()
            for seg in c_tree:
                start = seg.begin
                end = seg.end
                cns_a1 = []
                cns_a2 = []
                ccf_hat_a1 = []
                ccf_hat_a2 = []
                ccf_high_a1 = []
                ccf_high_a2 = []
                ccf_low_a1 = []
                ccf_low_a2 = []
                for i, sample in enumerate(self.sample_list):
                    seg_data = seg.data[i][1]
                    cns_a1.append(seg_data['cn_a1'])
                    cns_a2.append(seg_data['cn_a2'])
                    ccf_hat_a1.append(seg_data['ccf_hat_a1'] if seg_data['cn_a1'] != 1 else 0.)
                    ccf_hat_a2.append(seg_data['ccf_hat_a2'] if seg_data['cn_a2'] != 1 else 0.)
                    ccf_high_a1.append(seg_data['ccf_high_a1'] if seg_data['cn_a1'] != 1 else 0.)
                    ccf_high_a2.append(seg_data['ccf_high_a2'] if seg_data['cn_a2'] != 1 else 0.)
                    ccf_low_a1.append(seg_data['ccf_low_a1'] if seg_data['cn_a1'] != 1 else 0.)
                    ccf_low_a2.append(seg_data['ccf_low_a2'] if seg_data['cn_a2'] != 1 else 0.)
                cns_a1 = np.array(cns_a1)
                cns_a2 = np.array(cns_a2)
                if np.all(cns_a1 == 1):
                    pass
                elif np.all(cns_a1 >= 1) or np.all(cns_a1 <= 1):
                    if start < centromere < end:
                        event_segs.add((start, centromere, 'p', tuple(cns_a1), tuple(ccf_hat_a1), tuple(ccf_high_a1), tuple(ccf_low_a1), 'a1'))
                        event_segs.add((centromere, end, 'q', tuple(cns_a1), tuple(ccf_hat_a1), tuple(ccf_high_a1), tuple(ccf_low_a1), 'a1'))
                    elif end < centromere:
                        event_segs.add((start, end, 'p', tuple(cns_a1), tuple(ccf_hat_a1), tuple(ccf_high_a1), tuple(ccf_low_a1), 'a1'))
                    else:
                        event_segs.add((start, end, 'q', tuple(cns_a1), tuple(ccf_hat_a1), tuple(ccf_high_a1), tuple(ccf_low_a1), 'a1'))
                else:
                    logging.warning('Seg with inconsistent event: {}:{}:{}'.format(chrom, seg.begin, seg.end))
                if np.all(cns_a2 == 1):
                    pass
                elif np.all(cns_a2 >= 1) or np.all(cns_a2 <= 1):
                    if start < centromere < end:
                        event_segs.add((start, centromere, 'p', tuple(cns_a2), tuple(ccf_hat_a2), tuple(ccf_high_a2), tuple(ccf_low_a2), 'a2'))
                        event_segs.add((centromere, end, 'q', tuple(cns_a2), tuple(ccf_hat_a2), tuple(ccf_high_a2), tuple(ccf_low_a2), 'a2'))
                    elif end < centromere:
                        event_segs.add((start, end, 'p', tuple(cns_a2), tuple(ccf_hat_a2), tuple(ccf_high_a2), tuple(ccf_low_a2), 'a2'))
                    else:
                        event_segs.add((start, end, 'q', tuple(cns_a2), tuple(ccf_hat_a2), tuple(ccf_high_a2), tuple(ccf_low_a2), 'a2'))
                else:
                    logging.warning('Seg with inconsistent event: {}:{}:{}'.format(chrom, seg.begin, seg.end))
            neighbors = {s: set() for s in event_segs}
            for seg1, seg2 in itertools.combinations(event_segs, 2):
                s1_hat = np.array(seg1[4])
                s2_hat = np.array(seg2[4])
                if seg1[2] == seg2[2] and seg1[3] == seg2[3] and all(s1_hat >= np.array(seg2[6])) and all(s1_hat <= np.array(seg2[5])) \
                        and all(s2_hat >= np.array(seg1[6])) and all(s2_hat <= np.array(seg1[5])):
                    neighbors[seg1].add(seg2)
                    neighbors[seg2].add(seg1)

            def _BK(P, neighbors, R=frozenset(), X=frozenset()):
                if not P and not X:
                    yield R
                else:
                    for v in P:
                        for r in _BK(P & neighbors[v], neighbors, R=R | {v}, X=X & neighbors[v]):
                            yield r
                        P = P - {v}
                        X = X | {v}

            for clique in _BK(event_segs, neighbors):
                if clique:
                    clique_len = 0
                    clique_ccf_hat = np.zeros(n_samples)
                    clique_ccf_high = np.zeros(n_samples)
                    clique_ccf_low = np.zeros(n_samples)
                    n_segs = 0
                    for seg in clique:
                        clique_len += seg[1] - seg[0]
                        clique_ccf_hat += np.array(seg[4])
                        clique_ccf_high += np.array(seg[5])
                        clique_ccf_low += np.array(seg[6])
                        clique_arm = seg[2]
                        cn_category = 'Arm_gain' if all(np.array(seg[3]) > 1) else 'Arm_loss'
                        local_cn = seg[3]
                        n_segs += 1
                    clique_ccf_hat /= n_segs
                    clique_ccf_high /= n_segs
                    clique_ccf_low /= n_segs
                    arm_len = centromere if clique_arm == 'p' else csize - centromere
                    if clique_len > arm_len * .5:
                        self._add_cn_event_to_samples(chrom, 0, 0, clique_arm, local_cn, cn_category, clique_ccf_hat, clique_ccf_high,
                                                      clique_ccf_low)

    def _add_cn_event_to_samples(self, chrom, start, end, arm, cns, cn_category, ccf_hat, ccf_high, ccf_low):
        """
        Adds CN event to sample in low_coverage_mutations attr and mut hashtable

        """
        for i, sample in enumerate(self.sample_list):
            local_cn = cns[i]
            ccf_hat_i = ccf_hat[i] if local_cn != 1. else 0.
            ccf_high_i = ccf_high[i] if local_cn != 1. else 0.
            ccf_low_i = ccf_low[i] if local_cn != 1. else 0.
            cn = CopyNumberEvent(chrom, cn_category, start=start, end=end, ccf_hat=ccf_hat_i, ccf_high=ccf_high_i, ccf_low=ccf_low_i,
                                 local_cn=local_cn, from_sample=sample, arm=arm)
            sample.low_coverage_mutations.update({cn.var_str: cn})
            sample.add_muts_to_hashtable(cn)


#################################################################################
# NDHistogram() - helper class to store combined histograms of mutations to pass to DP
#################################################################################
class NDHistogram:
    """CLASS info

    FUNCTIONS:

    PROPERTIES:
    """

    def __init__(self, hist_array, labels, phasing=None, ignore_nan=False):
        conv = 1e-40

        # We need to noramlize the array to 1. This is hard.
        # This step does the following:
        # convert to log space after adding a small convolution paramter so we don't get INF and NAN
        # for each mutation
        # normalize the row to 1 in logspace.

        hist = np.asarray(hist_array, dtype=np.float32) + conv
        if (~(hist > 0)).any():
            logging.error("Negative histogram bin or NAN mutation!")
            if ignore_nan:
                logging.warning("Caught ignore nan flag, not exiting, setting nan values to zero")
                hist[np.logical_not(hist > 0)] = conv
            else:
                sys.exit(1)

        n_samples = np.shape(hist)[1]
        for sample in range(n_samples):
            hist[:, :, 0] = conv  ##set zero bins
        self._hist_array = np.apply_over_axes(lambda x, y: np.apply_along_axis(lambda z: z - logsumexp(z), y, x),
                                              np.log(hist), 2)
        ####

        self._labels = labels
        self._label_ids = dict([[y, x] for x, y in enumerate(labels)])
        self._phasing = {} if phasing is None else phasing
        self.n_samples = np.shape(hist_array)[1]
        self.n_bins = np.shape(hist_array)[-1]

    def __getitem__(self, key):
        return self._hist_array[self._label_ids[key]]

    def phase(self, m1, m2):
        return self._phasing[frozenset({m1, m2})]

    def iteritems(self):
        for idx, mut in enumerate(self._hist_array):
            yield self._labels[idx], mut

    @property
    def mutations(self):
        return {}


_cytoband_dict = {}
with open(os.path.dirname(__file__) + '/supplement_data/cytoBand.txt', 'r') as _f:
    for _i, _line in enumerate(_f):
        _row = _line.strip('\n').split('\t')
        _cytoband_dict[(_row[0], _row[3])] = _i

class Cytoband:
    band_nums = _cytoband_dict

    def __init__(self, chrom, band):
        self.chrom = chrom if chrom.startswith('chr') else 'chr' + chrom
        self.band = band

    def __sub__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot subtract Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            self_num = self.band_nums[(self.chrom, self.band)]
            other_num = self.band_nums[(other.chrom, other.band)]
            return self_num - other_num
        raise ValueError('Cannot subtract cytobands on different chromosomes')

    def __lt__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            return self - other < 0
        raise ValueError('Cannot compare cytobands on different chromosomes')

    def __gt__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            return self - other > 0
        raise ValueError('Cannot compare cytobands on different chromosomes')

    def __le__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            return self - other <= 0
        raise ValueError('Cannot compare cytobands on different chromosomes')

    def __ge__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            return self - other >= 0
        raise ValueError('Cannot compare cytobands on different chromosomes')

    def __eq__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.chrom, self.band))

    def __repr__(self):
        return str(self.chrom) + self.band

##############################################################

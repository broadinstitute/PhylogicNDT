# this is the phylogic engine module that is responsible for all computation and clustering
__version__ = "0.2.2"
# standard library
import collections
import logging

# statistics modules
import numpy as np

# modules from this PhylogicNDT package
from .DpEngine import DpEngine


class ClusterEngine:

    def __init__(self, patient):
        """

        Args:
            patient: data.Patient object

        Returns:
            None:
        """

        self.patient = patient
        """:type : data.Patient"""
        self.results = []
        self.common_mutations = collections.OrderedDict()
        self.common_subclones = collections.OrderedDict()

    def run_DP_ND(self, N_iter=5, PriorK=None, use_fixed=False, mode="tree"):

        if use_fixed:
            logging.error("Fixed mutations not configured for ND, proceed with caution.")

        if not PriorK:
            logging.error(
                "gamma prior mean and variance not specified, please specify --Pi_k_mu 10 and  --Pi_k_r 10, assuming values of 10.")
            PriorK = {'r': 10, 'mu': 10}  # standard values

        nd_hist = self.patient._make_ND_histogram()
        clustering = DpEngine(nd_hist, N_iter, PriorK)
        self.results = clustering.results
        self._ND_cluster_postprocess()  # Set cluster assignment, etc.
        self._ND_assign_setaside_mutations()

    def _ND_cluster_postprocess(self, clonal_cutoff=0.9):
        assign = self.results["assign"]

        for smpl_index, sample in enumerate(self.patient.sample_list):
            sample.DPresults_ND = self.results

            for mut_index, mut in enumerate(sample.concordant_variants):
                mut.cluster_assignment = assign[mut_index]

            for mut in sample.concordant_variants:
                mut.clust_ccf = np.argmax(self.results.clust_CCF_dens[mut.cluster_assignment - 1][smpl_index]) / 100.

            for mut in sample.concordant_variants:
                if mut.clust_ccf >= clonal_cutoff:
                    mut.clonality = 'clonal'
                elif mut.alt_cnt > 0:
                    mut.clonality = 'subclonal'
                else:
                    mut.clonality = 'not_detected'

    def _ND_assign_setaside_mutations(self):

        data = self.patient
        results = self.results
        grid_size = data.sample_list[0].ccf_grid_size

        # After run, assign unassigned mutations that have been removed for low coverage, and indels:
        # '''

        combined_lowcov = set()
        combined_cnvs = set()

        for sample in data.sample_list:

            for mut in sample.low_coverage_mutations.values():
                if mut.type == 'CNV':
                    combined_cnvs.add(mut)
                else:
                    combined_lowcov.add(mut)

        nclusters = len(results["clust_CCF_dens"])
        ccf_0 = np.zeros(len(data.sample_names) * grid_size)

        muts_added = 0
        combined_blacklist = set.union(*[set(x.artifacts_in_blacklist) for x in data.sample_list])
        for mut in combined_lowcov.union(combined_cnvs):
            if mut in combined_blacklist: continue  # skip blacklisted mutations!!
            mut_row = []
            if 'WGD' in mut.var_str:
                for sample in data.sample_list:
                    print
                    sample.WGD_status
                for sample in data.sample_list:
                    mut_row.append(sample.get_mut_by_varstr(mut.var_str))
            if len(mut_row) < len(data.sample_names):
                continue

            if mut in combined_lowcov:  # for lowcov/indels, assign based on likelihoods of joining clusters estimated from ccf_dist
                pvals = np.zeros((nclusters, len(data.sample_names)))
                for sample_index, ccf in enumerate([np.array(x.ccf_1d) for x in mut_row]):

                    for cluster_index, cluster in enumerate(results["clust_CCF_dens"]):
                        pvals[cluster_index][sample_index] = sum(
                            cluster[sample_index] / sum(cluster[sample_index]) * ccf)

                    data.sample_list[sample_index].concordant_variants.append(
                        data.sample_list[sample_index].get_mut_by_varstr(mut.var_str))
                pvals = np.prod(pvals, axis=1)
                mut_assignment = pvals.argmax() + 1
                self.results["assign"] = np.append(self.results["assign"], mut_assignment)
                self.results["var_loc"] = np.append(self.results["var_loc"], ccf_0)
                for sample_specific_mut in mut_row:
                    sample_specific_mut.cluster_assignment = mut_assignment
                muts_added += 1
            elif mut in combined_cnvs:  # for cnvs, assign based on point estimate of nearest clusters
                ccf_diffs = np.zeros((nclusters, len(data.sample_names)))
                for sample_index, ccf in enumerate([np.array(x.ccf_1d) for x in mut_row]):

                    for cluster_index, cluster in enumerate(results["clust_CCF_dens"]):
                        ccf_diffs[cluster_index][sample_index] = abs(
                            np.argmax(cluster[sample_index]) / float(grid_size - 1) - np.argmax(ccf) / float(
                                grid_size - 1))

                    data.sample_list[sample_index].concordant_variants.append(
                        data.sample_list[sample_index].get_mut_by_varstr(mut.var_str))
                summed_diffs = np.sum(ccf_diffs, axis=1)
                mut_assignment = summed_diffs.argmin() + 1
                max_diff_across_all_samples = np.max(ccf_diffs[summed_diffs.argmin()])
                if max_diff_across_all_samples > 0.5:
                    # don't assign if there is a time point where ccf difference at least 0.2
                    logging.info("Appending mut to blacklist since does not fit well with any cluster: " + mut.var_str)
                    for sample in data.sample_list:
                        sample.artifacts_in_blacklist.append(sample.get_mut_by_varstr(mut.var_str))
                        sample.get_mut_by_varstr(mut.var_str).blacklist_status = True
                        sample.known_blacklisted_mut.add(sample.get_mut_by_varstr(mut.var_str))
                    continue
                self.results["assign"] = np.append(self.results["assign"], mut_assignment)
                self.results["var_loc"] = np.append(self.results["var_loc"], ccf_0)
                for sample_specific_mut in mut_row:
                    sample_specific_mut.cluster_assignment = mut_assignment
                muts_added += 1

        # remove cn events in concordant variants which did not
        cnvs_to_remove = {}
        for sample in data.sample_list:
            cnvs_to_remove[sample] = []
            for mut in sample.artifacts_in_blacklist:
                cnvs_to_remove[sample].append(mut)
        for sample in data.sample_list:
            updated_concortdant_variants = []
            for mut in sample.concordant_variants:
                if mut in cnvs_to_remove[sample]:
                    continue
                else:
                    updated_concortdant_variants.append(mut)
            sample.concordant_variants = updated_concortdant_variants

        logging.info("Re-added {} low coverage mutations and indels, of {} that were removed.".format(muts_added, len(
            combined_lowcov.union(combined_cnvs))))

    def _build_common_sample_clone_table(self):
        data = self.patient
        results = self.results
        grid_size = data.sample_list[0].ccf_grid_size

        var_classes = collections.OrderedDict()
        tree_building_mutations = collections.OrderedDict()
        for smpl_index, sample in enumerate(data.sample_list):
            var_classes[smpl_index] = []
            for mut_index, mut in enumerate(sample.concordant_variants):
                if results["clust_CCF_dens"][results["assign"][mut_index] - 1][smpl_index].argmax() / float(
                        grid_size - 1) >= 0.95:
                    mut.cluster_assignment = results["assign"][mut_index]

                    var_classes[smpl_index].append("C")
                    # logging.info( 'adding mutation as private clonal '+str(mut.var_str))
                    if mut not in self.common_mutations:
                        self.common_mutations[mut] = [0] * (
                                len(data.sample_list) + 1)  # init as zeros for all including normal
                    self.common_mutations[mut][smpl_index] = 1
                else:
                    var_classes[smpl_index].append("S")

                if results["clust_CCF_dens"][results["assign"][mut_index] - 1][smpl_index].argmax() / float(
                        grid_size - 1) >= 0.90:
                    # logging.info( 'adding mutation as private clonal '+str(mut.var_str))
                    if mut not in tree_building_mutations:
                        tree_building_mutations[mut] = [0] * (
                                len(data.sample_list) + 1)  # init as zeros for all including normal
                    tree_building_mutations[mut][smpl_index] = 1
                elif results["clust_CCF_dens"][results["assign"][mut_index] - 1][smpl_index].argmax() / float(
                        grid_size - 1) >= 0.25:
                    if mut not in tree_building_mutations:
                        tree_building_mutations[mut] = [0] * (
                                len(data.sample_list) + 1)  # init as zeros for all including normal
                    tree_building_mutations[mut][
                        smpl_index] = 0  # .5  # results["clust_CCF_dens"][results["assign"][mut_index]-1][smpl_index].argmax()/float(grid_size-1)

        for smpl_index, sample in enumerate(data.sample_list):
            for mut_index, mut in enumerate(sample.concordant_variants):
                if mut.alt_cnt is 0 or results["clust_CCF_dens"][results["assign"][mut_index] - 1][
                    smpl_index].argmax() / float(grid_size - 1) < 0.05:
                    continue
                mut.cluster_assignment = results["assign"][mut_index]
                if mut in self.common_mutations:
                    if self.common_mutations[mut][smpl_index] < 1:  # If it's clonal in a sample other than this one.
                        self.common_mutations[mut][smpl_index] = 0.5
                    else:
                        continue
                elif mut not in self.common_mutations:
                    if mut not in self.common_subclones:
                        self.common_subclones[mut] = [0] * (
                                len(data.sample_list) + 1)  # init as zeros for all including normal
                    self.common_subclones[mut][smpl_index] = 0.5

        return tree_building_mutations

    def _init_clust_CCF_prior(self, grid_length):
        prior_size = grid_length ** len(self.patient.sample_list)
        clust_CCF_prior = [1.0 / prior_size] * (prior_size)  # make list grid_size^2 with normalized
        return clust_CCF_prior

    def _init_clust_CCF_prior_1D(self, grid_length):

        prior_size = grid_length
        clust_CCF_prior = [1.0 / prior_size] * (prior_size)  # make list grid_size^2 with normalized
        return clust_CCF_prior

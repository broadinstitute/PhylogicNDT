import numpy as np
import logging
import random
import bisect
import itertools
import scipy.stats

from scipy.stats import linregress
# Logsumexp options
from scipy.misc import logsumexp as logsumexp_scipy  # always import as double safe method (slow)

import pkgutil

if pkgutil.find_loader('sselogsumexp') is not None:
    logging.info("Using fast logsumexp")
    from sselogsumexp import logsumexp
else:
    logging.info("Using scipy (slower) logsumexp")
    from scipy.misc import logsumexp

from BuildTree import ShuffleMutations
from BuildTree.CellPopulationEngine import CellPopulationEngine


class GrowthKineticsEngine:

    def __init__(self, patient, wbc):
        # patient object should have pointer to clustering results
        self._patient = patient
        self._cp_engine = CellPopulationEngine(self._patient)
        if patient.ClusteringResults:
            self._clustering_results = patient.ClusteringResults
        else:
            logging.error(' Clustering results are not found, need to run clustering first')

        self._wbc = wbc
        self._indiv = self._patient.indiv_name
        self.n_iter = None
        self.time_points = self._clustering_results.samples
        self.grid = np.linspace(0, 1, 101)

        wbc = np.array(self._wbc)  # wbc used to be multiplied by the factor: *10**9/1000.

        n_clusters = None

        original_wbc = np.array(wbc[:])

    def one_iteration_fix_k(self):

        random.shuffle(self.mutations)
        for mut in self.mutations:

            loglik = np.ones((self.n_clusters, self.n_samples), dtype=np.float64) * -np.inf
            const_array = np.zeros((self.n_clusters,), dtype=np.float64)

            if len(mut.assigned_to) == 1:
                continue  # don't reassign last mutation

            for cluster_idx, cluster in enumerate(self.clusterlist):
                # if the current point is the only thing in the cluster...
                # This seems to work empirically (as well as theoretically)
                if len(cluster) == 1 and mut.assigned_to == cluster:
                    continue
                stay_in_clust_const = len(cluster) / float(self.n_muts - 1 + self.alpha)

                const_array[cluster_idx] = np.log(stay_in_clust_const)

                if mut not in cluster:  # TODO: redefine in
                    loglik[cluster_idx] = self.logsum_of_marginals_per_sample(cluster.normed_hist + mut.loghist)
                else:
                    loglik[cluster_idx] = self.logsum_of_marginals_per_sample(
                        self.normalize_loghist_with_prior(cluster - mut) + mut.loghist)

            loglik = np.sum(loglik, axis=1)  # + const_array

            loglik = loglik - logsumexp_scipy(loglik)
            loglik = loglik + const_array

            c_lik = np.exp(loglik - logsumexp_scipy(loglik))

            # if np.random.random() < 0.1: print sum(c_lik[:-1])
            new_cluster_idx = np.nonzero(np.random.multinomial(1, c_lik) == 1)[0][0]
            if new_cluster_idx == self.n_clusters:  # new cluster
                mut.assigned_to -= mut
                DP_cluster(self, mut)  # create new cluster, lists updated automatically
            else:
                new_cluster = self.clusterlist[new_cluster_idx]
                mut.assigned_to -= mut
                new_cluster += mut

        cluster_counter = itertools.count()
        next(cluster_counter)
        real_index = dict([[x.id, next(cluster_counter)] for x in self.clusterlist])
        self.results.assign.append([real_index[x.assigned_to.id] for x in self.mutations])
        self.results.alpha.append(self.alpha)
        self.results.eta.append(self.eta)
        self.results.cluster_loghistograms.append([cluster.normed_hist for cluster in self.clusterlist])
        self.results.cluster_positions.append(
            [[np.argmax(x) for x in cluster.normed_hist] for cluster in self.clusterlist])
        self.results.clust_prop.append([len(cluster) / float(self.n_muts) for cluster in self.clusterlist])
        self.results.clust_size.append([len(cluster) for cluster in self.clusterlist])
        self.results.K.append(self.n_clusters)

        return [real_index[x.assigned_to.id] for x in self.mutations], [cluster.normed_hist for cluster in
                                                                        self.clusterlist]

    @staticmethod
    def diff_ccf_uneven(ccf1, ccf2):
        """ Histogram of CCF1-CCF2 """

        ccf_dist1 = np.append(ccf1, [0] * len(ccf1))
        ccf_dist2 = np.append(ccf2, [0] * len(ccf1))

        convoluted_dist = []
        for k in range(len(ccf1)):
            inner_product = np.inner(ccf_dist1[0:len(ccf1)], ccf_dist2[len(ccf1) - 1 - k:2 * len(ccf1) - 1 - k])
            convoluted_dist.append(inner_product)
        for k in range(1, len(ccf1)):
            inner_product = np.inner(ccf_dist2[0:len(ccf1)], ccf_dist1[k:len(ccf1) + k])
            convoluted_dist.append(inner_product)
        return np.array(convoluted_dist[len(ccf1) - 1:]) / float(sum(convoluted_dist))

    def get_phylo_adj_ccf_dist(self, cluster_dens, tree):
        cluster_dens = np.exp(cluster_dens)
        cluster_dens_adj = np.swapaxes(cluster_dens, 0, 1).tolist()
        cluster_dens_org = np.swapaxes(cluster_dens, 0, 1).tolist()

        for c_idx, cluster in enumerate(np.swapaxes(cluster_dens, 0, 1)):
            for child in self._patient.TopTree.successors(c_idx):
                for tp in range(len_pre_tp):
                    cluster_dens_adj[c_idx][tp] = self.diff_ccf_uneven(cluster_dens_adj[c_idx][tp],
                                                                       cluster_dens_org[child][tp])

        return np.swapaxes(np.array(cluster_dens_adj), 0, 1)

    """
    @staticmethod
    def emd_nd(u, v):
        tot = 0
        U = np.reshape(u, (-1, 101))
        V = np.reshape(v, (-1, 101))
        for s in zip(U, V):
            tot += emd(np.atleast_2d(s[0]), np.atleast_2d(s[1]))
        return tot

    def fix_labels_v2(self, cluster_densities, orig_dens_tp, n_clusters):
        dist = []
        cluster_dens_tp = np.swapaxes(cluster_densities, 0, 1)
        for c_n_o, cluster_old in enumerate(orig_dens_tp):
            if c_n_o == 0:
                continue
            for c_n_i, cluster_new in enumerate(cluster_dens_tp):
                if c_n_i == 0:
                    continue
                dist.append([self.emd_nd(cluster_old, cluster_new), c_n_o, c_n_i])

        assigned_new = set([0])
        assigned_old = set([0])
        mapping = {0: 0}

        for distance, c_n_o, c_n_i in sorted(dist, key=lambda x: x[0]):
            if c_n_o in assigned_old:
                continue
            if c_n_i in assigned_new:
                continue

            mapping[c_n_o] = c_n_i
            assigned_old.add(c_n_o)
            assigned_new.add(c_n_i)

        new_densities = []
        for c_n_o in range(n_clusters):
            new_densities.append(cluster_dens_tp[mapping[c_n_o]])

        return np.swapaxes(np.array(new_densities), 0, 1)

    """

    def line_fit(self, x, c_idx, fb_x_vals, len_pre_tp, adj_dens):
        # TODO: understand this function
        """ """
        slope, intercept = x
        y_domain = [np.log(self.grid * self._wbc[tp_idx] + 1e-40) for tp_idx in range(len_pre_tp)]
        y_weights = [adj_dens[tp_idx][c_idx] for tp_idx in range(len_pre_tp)]
        line_y_vals = slope * fb_x_vals + intercept

        selected_weight = [
            min(
                sum(y_weights[tp_idx][:min(bisect.bisect(y_domain[tp_idx], line_y_vals[tp_idx]), 100) + 1]),
                sum(y_weights[tp_idx][min(bisect.bisect(y_domain[tp_idx], line_y_vals[tp_idx]), 100):])
            )
            for tp_idx in range(len_pre_tp)]

        return selected_weight

    def line_fit_pval(self, x, c_idx, fb_x_vals, len_pre_tp, adj_dens):
        return min(self.line_fit(x, c_idx, fb_x_vals, len_pre_tp, adj_dens))

    def line_fit_err(self, x, c_idx, fb_x_vals, len_pre_tp, adj_dens):
        return -sum(self.line_fit(x, c_idx, fb_x_vals, len_pre_tp, adj_dens))

    def run(self, times=None, n_iter=10):
        """ """
        # Number of samples for this Patient

        time_points = self._patient.ClusteringResults.sample_list
        original_wbc = np.array(self.wbc[:])
        n_clusters = len(self._patient.ClusteringResults)

        cluster_rates = {x: [] for x in range(n_clusters)}
        cluster_rates_diff = {x: [] for x in range(n_clusters)}
        cluster_line_fit_min_lik = {x: [] for x in range(n_clusters)}

        # If times of samples are not provided, treat as an integers
        if not times:
            times = np.array(range(len(time_points))) + 1

        try:
            # TODO: searches for first negative time? Is it pre-treatment
            len_pre_tp = list(np.sign(times)).index(-1)
        except:
            # TODO: Otherwise looks at all times
            len_pre_tp = len(times)

        for n in range(n_iter):

            # sample_wbc(wbc_ranges)
            # TODO: do not understand why multiplying by random
            wbc = original_wbc * (1 + np.array([(np.random.random() - 0.5) / 100. for x in range(len(original_wbc))]))

            # Shuffle mutations
            ShuffleMutations.shuffling(self._patient.ClusteringResults, self._patient.sample_list)
            # Identify cluster label switching after mutation shuffling
            ShuffleMutations.fix_cluster_lables(self._patient.ClusteringResults)

            # Computing constrained ccf distribution
            constrained_ccf = self._cp_engine.compute_constrained_ccf(n_iter=1)
            # Adjust constrained ccf according to phylogenetic tree
            cell_abundance = self._cp_engine.get_cell_abundance(constrained_ccf)

            # Loop where all the rates are calculated via linear fit to the log data.
            cluster_rates_iter = {}
            for cluster_id, cluster in self._patient.ClusteringResult.items():
                cluster_rates_iter[cluster_id] = linregress(times, np.log(cluster._hist * wbc)).slope
                cluster_rates[cluster_id].append(cluster_rates_iter[cluster_id])

            fb_x_vals = np.array(times[:len_pre_tp])

            for cluster_id, cluster in self._patient.ClusteringResult.items():
                if not cluster.blacklisted:
                    cluster_abundance = []
                    for sample, sample_abundances in cell_abundance.items():
                        cluster_abundance.append(sample_abundances[cluster_id])
                    cluster_abundance = np.asarray(cluster_abundance, dtype=np.float32)
                    cluster_abundance = np.log(cluster_abundance, dtype=np.float32) + 0.001

                    # Calculate a linear least-squares regression for two sets of measurements.
                    # Both arrays should have the same length.
                    fit_res = linregress(fb_x_vals, (cluster_abundance + np.log(wbc)))
                    # Minimization of scalar function of one or more variables.
                    slope_f, intercept_f = scipy.optimize.minimize(self.line_fit_err,
                                                                   [fit_res.slope, fit_res.intercept],
                                                                   method="Nelder-Mead").x
                    p_val_cut = self.line_fit_pval([slope_f, intercept_f])

                    cluster_line_fit_min_lik[cluster_id].append(p_val_cut)

            for cluster_id, rate in cluster_rates_iter.items():
                try:
                    parent = self._patient.TopTree.nodes[cluster_id]
                    diff_rate = rate - cluster_rates_iter[parent.identifier]
                except:
                    diff_rate = 0
                cluster_rates_diff[cluster_id].append(diff_rate)

    """
    def plot_growth_kinetics(self):
        import matplotlib as plt
        for c_idx, cluster in enumerate(clusters_.T):
            plt.plot(times, cluster, color=phylo_cmap.colors[c_idx], alpha=0.1)

    def plot_delta_rates(cluster_rates_diff):
        plt.figure()

        out_delta_rates = open(indiv + '.gr_delta.tsv', "w")
        for clust, rate in cluster_rates_diff.items()[1:]:
            if sum(rate) == 0: 
                continue
            sns.distplot(np.array(rate), bins=35,
                         label=str(clust + 1) + " - %1.3f" % (sum(np.array(rate) < 0) / float(len(rate))),
                         color=phylo_cmap.colors[clust])
            out_delta_rates.write("Cluster " + str(clust) + "\n")
            out_delta_rates.write("[" + ",".join([str(x) for x in rate]) + "]\n")
        out_delta_rates.close()
        plt.title("Difference to Parent GR")
        plt.xlabel("delta GR")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.savefig(indiv + ".Detla_GR.pdf")

    def write_growth_kinetics_file(self):
        # File to record the raw cell counts that the algorithm predicts for each iteration
        indiv = self._patient.indiv
        raw_growth_file = open(indiv + '.raw_growth.txt', "w")
        raw_growth_file.write("Iter:" + str(n) + "\n")
        raw_growth_file.write("Cluster:" + str(c_idx) + "\n")
        raw_growth_file.write("[" + ",".join([str(x) for x in log_growth]) + "]\n")
        raw_growth_file.close()


     def line_fit_pval(self, x, c_idx, fb_x_vals, len_pre_tp, adj_dens):
        slope, intercept = x
        y_domain = [np.log(self.grid * self._wbc[tp_idx] + 1e-40) for tp_idx in range(len_pre_tp)]
        y_weights = [adj_dens[tp_idx][c_idx] for tp_idx in range(len_pre_tp)]

        line_y_vals = slope * fb_x_vals + intercept

        selected_weight = [
            min(
                sum(y_weights[tp_idx][:min(bisect.bisect(y_domain[tp_idx], line_y_vals[tp_idx]), 100) + 1]),
                sum(y_weights[tp_idx][min(bisect.bisect(y_domain[tp_idx], line_y_vals[tp_idx]), 100):])
            )
            for tp_idx in range(len_pre_tp)]

        return min(selected_weight)

     def assign_by_dp(self):
        assignments, cluster_densities = self.one_iteration_fix_k()
        cluster_densities = np.swapaxes(np.array(cluster_densities), 0, 1)
        new_clusters = {}
        for c_idx, mutation in zip(assignments, clustering_engine.mutations):
            if c_idx not in new_clusters:
                new_clusters[c_idx] = set()
            try:
                new_clusters[c_idx].add(var_ccfs.sample_list[0].get_mut_by_varstr(mutation.id))
            except:
                pass

        return cluster_densities, new_clusters, assignments

    def write_tsv(cluster_rates, cluster_rates_diff):
        out_tsv = open(indiv + '.gr.tsv', "w")
        out_tsv.write("indiv\tclust\tGR\tdelta_GR_Parent\tp_value\n")
        clusters = {}
        d = {}
        for cluster_idx, cluster in cluster_rates.items():
            clusters[cluster_idx] = [np.nanmedian(np.array(cluster)[n_iter // 10:])]
        for cluster_idx, cluster in cluster_rates_diff.items():
            clusters[cluster_idx].append(np.nanmedian(np.array(cluster)[n_iter // 10:]) if cluster_idx > 0 else "NA")
        for cluster_idx, cluster in cluster_rates_diff.items():
            clusters[cluster_idx].append(np.nansum(np.array(cluster)[n_iter // 10:] < 0) / len(
                np.array(cluster)[n_iter // 10:]) if cluster_idx > 0 else "NA")

        for cluster, values in clusters.items():

            if values[2] == 1: values[2] = 1 - 1 / n_iter
            if values[2] == 0: values[2] = 1 / n_iter

            out_tsv.write("\t".join([indiv, str(cluster), str(values[0]), str(values[1]), str(values[2])]) + "\n")

        out_tsv.close()


    """

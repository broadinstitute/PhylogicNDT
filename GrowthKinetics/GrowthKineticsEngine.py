import numpy as np
import logging
import random
import bisect
import itertools
import scipy.stats
from emd import emd
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


class GrowthKineticsEngine:

    def __init__(self, patient, wbc):
        # patient object should have pointer to clustering results object but for now it is two separate objects
        self._patient = patient
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

            if len(mut.assigned_to) == 1: continue  # don't reassign last mutation

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

    #########################################################################
    #  Functions of reshuffling mutations #
    #########################################################################
    @staticmethod
    def logsum_of_marginals_per_sample(loghist):
        return np.apply_along_axis(lambda x: logsumexp_scipy(x), 1, np.array(loghist, dtype=np.float32))

    @staticmethod
    def make_nd_histogram(hist_array):
        conv = 1e-40
        hist = np.asarray(hist_array, dtype=np.float32) + conv
        n_samples = np.shape(hist)[1]
        for i in range(n_samples):
            hist[:, :, 0] = conv
        return np.apply_over_axes(lambda x, y: np.apply_along_axis(lambda z: z - logsumexp_scipy(z), y, x),
                                  np.log(hist), 2)

    def reshuffle_mutations(self, cluster_ccf, mut_ccf):
        combined_ccf = []
        # TODO: for each mutation
        for mutations in mut_ccf:
            # Get all ccf distribution from each time point
            mut_nd_ccf = [ccf_distrib_1, ccf_distrib_2]
            combined_ccf.append(mut_nd_ccf)
        nd_histogram = self.make_nd_histogram(combined_ccf)

        # TODO: need to keep track of mutation ids in nd_histogram (somehow)
        # compute probability mutation belong to cluster
        # sample from this probability, pick new cluster (or old)
        # update cluster density

    @staticmethod
    def logsum_of_marginals_per_sample(loghist):
        return np.apply_along_axis(lambda x: logsumexp(x), 1, np.array(loghist, dtype=np.float32))

    def one_iteration_fix_k(self, mutations, nd_histograms, clusters_ccf, clustering_results):

        n_clusters = len(clusters_ccf)
        n_samples = np.shape(nd_histograms)[1]
        for var_str in mutations:

            idx = mutations[var_str]
            loglik = np.ones((n_clusters, n_samples), dtype=np.float64) * -np.inf
            const_array = np.zeros((n_clusters,), dtype=np.float64)

            for cluster_id, cluster_ccf in clusters_ccf.items():
                # print (logsum_of_marginals_per_sample(clusters_ccf[cluster_id] + nd_histograms[idx]))
                loglik[cluster_id] = self.logsum_of_marginals_per_sample(cluster_ccf + nd_histograms[idx])

            loglik = np.sum(loglik, axis=1)  # + const_array
            loglik = loglik - logsumexp_scipy(loglik)
            loglik = loglik + const_array

            c_lik = np.exp(loglik - logsumexp_scipy(loglik))
            new_cluster_idx = np.argmax(c_lik)
            # new_cluster_idx = np.nonzero(np.random.multinomial(1, c_lik) == 1)[0][0]
            if var_str not in clustering_results:
                clustering_results[var_str] = []
            clustering_results[var_str].append(new_cluster_idx)
        return clustering_results

    def line_fit_err(self, x, c_idx, fb_x_vals, len_pre_tp, adj_dens):
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

        return -sum(selected_weight)

    def run(self, times, n_iter=10):
        """ """
        # Number of samples for this Patient
        time_points = self._clustering_results.samples
        original_wbc = np.array(self.wbc[:])

        try:
            len_pre_tp = list(np.sign(times)).index(-1)
        except:
            len_pre_tp = len(times)

        for n in range(n_iter):

            # sample_wbc(wbc_ranges)
            wbc = original_wbc * (1 + np.array([(np.random.random() - 0.5) / 100. for x in range(len(original_wbc))]))

            cluster_densities, new_clusters, assignments = self.assign_by_dp()

            cluster_densities = self.fix_labels_v2(cluster_densities,
                                                   var_ccfs.sample_list[0].DPresults_ND["clust_CCF_dens"],
                                                   n_clusters, n_timepoints)

            cluster_densities = np.swapaxes(np.array(cluster_densities), 0, 1)
            cluster_densities[0] = clonal_dens
            cluster_densities = np.swapaxes(cluster_densities, 0, 1)

            parents, children, siblings, tree_obj = re_sample_tree(cluster_densities, parents, n_clusters, n_timepoints)

            # Returns an adjusted probability density (?) where the children are substracted from the parents
            adj_dens = self.get_phylo_adj_ccf_dist(cluster_densities, tree_obj)

            # Draws the cluster ccfs (?) given the adjusted probability density
            clusters = draw_new_cluster_pos(np.log(adj_dens), clusters, parents, children, siblings, n_clusters,
                                            n_timepoints)

            current_tree_edges = frozenset(tree_obj.edges())  # just so we don't have to write this out over and over
            if current_tree_edges not in trees_by_edge:
                trees_by_edge[current_tree_edges] = []
            trees_by_edge[current_tree_edges].append(tree_obj.calc_pos_lik())

            # Loop where all the rates are claculated via linear fit to the log data.
            rates_by_cluster = {}
            for c_idx, cluster in enumerate(np.clip(clusters.T, 0.01, 1)):
                tpoints = times[:len_pre_tp]
                log_growth = np.log(cluster * wbc)[:len_pre_tp]

                rates_by_cluster[c_idx] = linregress(times[:len_pre_tp], np.log(cluster * wbc)[:len_pre_tp]).slope
                cluster_rates[c_idx].append(rates_by_cluster[c_idx])

            fb_x_vals = np.array(times[:len_pre_tp])

            for c_idx in range(n_clusters):
                if c_idx in rm_clusters:
                    continue

                fit_res = linregress(fb_x_vals,
                                     ((np.log(np.argmax(adj_dens, axis=2) / 100. + 0.001).T[c_idx]) + np.log(wbc))[
                                     :len_pre_tp])
                slope_f, intercept_f = scipy.optimize.minimize(line_fit_err, [fit_res.slope, fit_res.intercept],
                                                               method="Nelder-Mead").x
                p_val_cut = line_fit_pval([slope_f, intercept_f])

                cluster_line_fit_min_lik[c_idx].append(p_val_cut)

            for c_idx, rate in rates_by_cluster.items():
                try:
                    diff_rate = rate - rates_by_cluster[tree_obj.predecessors(c_idx)[0]]
                except:
                    diff_rate = 0
                cluster_rates_diff[c_idx].append(diff_rate)

            clusters_ = draw_new_cluster_pos(cluster_densities, clusters, parents, children, siblings, n_clusters,
                                             n_timepoints)
            c_log.append(cluster_densities)

    def plot_growth_kinetics(self):
        import matplotlib as plt
        for c_idx, cluster in enumerate(clusters_.T):
            plt.plot(times, cluster, color=phylo_cmap.colors[c_idx], alpha=0.1)

    def write_growth_kinetics_file(self):
        """ File to record the raw cell counts that the algorithm predicts for each iteration """
        indiv = self._patient.indiv
        raw_growth_file = open(indiv + '.raw_growth.txt', "w")
        raw_growth_file.write("Iter:" + str(n) + "\n")
        raw_growth_file.write("Cluster:" + str(c_idx) + "\n")
        raw_growth_file.write("[" + ",".join([str(x) for x in log_growth]) + "]\n")
        raw_growth_file.close()

import numpy as np
import bisect
from collections import defaultdict
from scipy.stats import linregress
from scipy.interpolate import interp1d


class GrowthKineticsEngine:

    def __init__(self, patient, wbc):
        self._patient = patient
        self._wbc = np.array(wbc)
        self._growth_rates = defaultdict(list)

    @property
    def growth_rates(self):
        return self._growth_rates

    @property
    def wbc(self):
        return self._wbc

    def estimate_growth_rate(self, mcmc_trace_cell_abundance, sample_time_points, wbc_time_points=None, n_iter=100,
                             conv=1e-40):
        '''

        '''
        # Number of samples for this Patient
        sample_list = sorted(sample_time_points, key=sample_time_points.__getitem__)
        sample_time_points_arr = np.array([sample_time_points[s] for s in sample_list])
        n_clusters = len(list(mcmc_trace_cell_abundance[sample_list[0]]))
        # cluster_rates = defaultdict(list)
        # If times of samples are not provided, treat as an integers
        if wbc_time_points is None:
            assert len(self.wbc) == len(sample_list), 'WBCs not corresponding to samples need time points'
            wbc_time_points = sample_time_points_arr
        cond = (wbc_time_points >= min(sample_time_points_arr)) & (wbc_time_points <= max(sample_time_points_arr))
        inner_wbc = self.wbc[cond]
        inner_wbc_time_points = wbc_time_points[cond]
        for n in range(n_iter):
            adj_wbc = inner_wbc * (1 + (np.random.random(len(inner_wbc)) - .5) / 100.)
            for cluster_id in range(1, n_clusters + 1):
                cluster_abundances = [mcmc_trace_cell_abundance[sample_id][cluster_id][n] + conv for sample_id in
                                      sample_list]
                inner_cluster_abundances = interp1d(sample_time_points_arr, cluster_abundances)(inner_wbc_time_points)
                cluster_slope = linregress(inner_wbc_time_points, inner_cluster_abundances * adj_wbc).slope
                self._growth_rates[cluster_id].append(cluster_slope)

    def line_fit(self, x, c_idx, fb_x_vals, len_pre_tp, adj_dens):
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

    def line_fit_err(self, x, c_idx, wbc, fb_x_vals, len_pre_tp, adj_dens):
        slope, intercept = x
        grid = np.arange(101)
        y_domain = [np.log(grid * wbc[tp_idx] + 1e-40) for tp_idx in range(len_pre_tp)]
        y_weights = [adj_dens[tp_idx][c_idx] for tp_idx in range(len_pre_tp)]

        line_y_vals = slope * fb_x_vals + intercept

        selected_weight = []
        for tp_idx in range(len_pre_tp):
            sum0 = sum(y_weights[tp_idx][min(bisect.bisect(y_domain[tp_idx], line_y_vals[tp_idx]), 100):])
            sum1 = sum(y_weights[tp_idx][:min(bisect.bisect(y_domain[tp_idx], line_y_vals[tp_idx]), 100) + 1])
            selected_weight.append(min(sum0, sum1))

        return -sum(selected_weight)

    def line_fit_pval(self, x, c_idx, wbc, fb_x_vals, len_pre_tp, adj_dens):
        slope, intercept = x
        grid = np.arange(101)
        y_domain = [np.log(grid * wbc[tp_idx] + 1e-40) for tp_idx in range(len_pre_tp)]
        y_weights = [adj_dens[tp_idx][c_idx] for tp_idx in range(len_pre_tp)]

        line_y_vals = slope * fb_x_vals + intercept

        selected_weight = [
            min(
                sum(y_weights[tp_idx][:min(bisect.bisect(y_domain[tp_idx], line_y_vals[tp_idx]), 100) + 1]),
                sum(y_weights[tp_idx][min(bisect.bisect(y_domain[tp_idx], line_y_vals[tp_idx]), 100):])
            )
            for tp_idx in range(len_pre_tp)]

        return min(selected_weight)

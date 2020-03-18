import numpy as np
import scipy.stats


def calc_ccf(local_cn_a1, local_cn_a2, alt_cnt, ref_cnt, purity, grid_size=101):
    """
    Calculate CCF from local copy number, alt/ref count, and purity
    Args:
        local_cn_a1: Minor allele local copy number
        local_cn_a2: Major allele local copy number
        alt_cnt: Alt count
        ref_cnt: Ref count
        purity: Tumor fraction
        grid_size: number of bins

    Returns:
        numpy array representing the CCF histogram
    """
    if local_cn_a1 > 1.:
        clonal_cn_a1 = np.floor(local_cn_a1)
        subclonal_cn_a1 = np.ceil(local_cn_a1)
        subclonal_frac_a1 = local_cn_a1 - clonal_cn_a1
    else:
        clonal_cn_a1 = np.ceil(local_cn_a1)
        subclonal_cn_a1 = np.floor(local_cn_a1)
        subclonal_frac_a1 = clonal_cn_a1 - local_cn_a1

    if local_cn_a2 > 1.:
        clonal_cn_a2 = np.floor(local_cn_a2)
        subclonal_cn_a2 = np.ceil(local_cn_a2)
        subclonal_frac_a2 = local_cn_a2 - clonal_cn_a2
    else:
        clonal_cn_a2 = np.ceil(local_cn_a2)
        subclonal_cn_a2 = np.floor(local_cn_a2)
        subclonal_frac_a2 = clonal_cn_a2 - local_cn_a2

    total_cov = alt_cnt + ref_cnt
    total_cn = local_cn_a1 + local_cn_a2

    # Calculate likelihood of clonality and subclonality
    ccf_dist_m1 = ccf_dist_from_params(1, total_cn, alt_cnt, ref_cnt, purity, grid_size=grid_size)
    ccf_mode = np.argmax(ccf_dist_m1) / 100.
    af_mode = ccf_mode * purity / (total_cn * purity + 2 * (1 - purity))
    p_subclonal = scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode)
    p_clonal = 0
    # Add likelihood of each integer multiplicity
    for mult in np.arange(2, clonal_cn_a2 + 1):
        af_mode = mult * purity / (total_cn * purity + 2 * (1 - purity))
        mult_weight = scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode)
        p_clonal += mult_weight
    # Add likelihood of subclonal shifts in multiplicity > 1 due to gains/deletions
    if subclonal_frac_a1:
        for mult in np.arange(subclonal_frac_a1 + 1, subclonal_cn_a1):
            af_mode = mult * purity / (total_cn * purity + 2 * (1 - purity))
            mult_weight = scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode)
            p_clonal += mult_weight
    if subclonal_frac_a2:
        for mult in np.arange(subclonal_frac_a2 + 1, subclonal_cn_a2):
            af_mode = mult * purity / (total_cn * purity + 2 * (1 - purity))
            mult_weight = scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode)
            p_clonal += mult_weight

    # Calculate ccf for multiplicity 1
    subc_ccf_hist = np.zeros(grid_size)
    if local_cn_a2 >= 1.:
        # Add mult1 ccf dist weighted by maximum subclonal likelihood
        subc_ccf_hist += ccf_dist_m1 * p_subclonal
    # If subclonality is from subclonal deletion, cut off CCF at subclonal fraction and weight by CCF likelihood
    if subclonal_frac_a1:
        subc_ccf_dist1 = ccf_dist_m1.copy()
        subc_ccf_dist2 = ccf_dist_m1.copy()
        subc_ccf_dist1[int(subclonal_frac_a1 * 100):] = 0
        subc_ccf_dist2[int((1 - subclonal_frac_a1) * 100):] = 0
        sum_dist_1 = sum(subc_ccf_dist1)
        if sum_dist_1:
            subc_ccf_dist1 /= sum_dist_1
            af_mode_1 = subclonal_frac_a1 * purity / (total_cn * purity + 2 * (1 - purity))
            subc_ccf_hist += subc_ccf_dist1 * scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode_1)
        sum_dist_2 = sum(subc_ccf_dist2)
        if sum_dist_2:
            subc_ccf_dist2 /= sum_dist_2
            af_mode_2 = (1 - subclonal_frac_a1) * purity / (total_cn * purity + 2 * (1 - purity))
            subc_ccf_hist += subc_ccf_dist2 * scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode_2)
    if subclonal_frac_a2:
        subc_ccf_dist1 = ccf_dist_m1.copy()
        subc_ccf_dist2 = ccf_dist_m1.copy()
        subc_ccf_dist1[int(subclonal_frac_a2 * 100):] = 0
        subc_ccf_dist2[int((1 - subclonal_frac_a2) * 100)] = 0
        sum_dist_1 = sum(subc_ccf_dist1)
        if sum_dist_1:
            subc_ccf_dist1 /= sum_dist_1
            af_mode_1 = subclonal_frac_a2 * purity / (total_cn * purity + 2 * (1 - purity))
            subc_ccf_hist += subc_ccf_dist1 * scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode_1)
        sum_dist_2 = sum(subc_ccf_dist2)
        if sum_dist_2:
            subc_ccf_dist2 /= sum_dist_2
            af_mode_2 = (1 - subclonal_frac_a2) * purity / (total_cn * purity + 2 * (1 - purity))
            subc_ccf_hist += subc_ccf_dist2 * scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode_2)
    subc_ccf_hist /= sum(subc_ccf_hist)

    bins = np.linspace(0, 1, grid_size)
    clonal_ccf_hist = scipy.stats.beta.pdf(bins, alt_cnt + 1, 1)
    clonal_ccf_hist /= sum(clonal_ccf_hist)

    ccf_hist = p_subclonal * subc_ccf_hist + p_clonal * clonal_ccf_hist
    ccf_hist /= sum(ccf_hist)

    return ccf_hist


def ccf_dist_from_params(mult, total_cn, alt_cnt, ref_cnt, purity, grid_size=101):
    """
    Calculate a ccf distribution for a given multiplicity
    Args:
        mult: multiplicity
        total_cn: total local copy number
        alt_cnt: alt count
        ref_cnt: ref count
        purity: tumor fraction
        grid_size: number of bins

    Returns:
        CCF distribution and mean ccf for given multiplicity
    """
    if mult == 0:
        ccf_dist = np.zeros(grid_size)
        ccf_dist[0] = 1.
        return ccf_dist, 0.

    ccf_bins = np.linspace(0, 1, grid_size)

    # Since transformation is linear, ToV formula not necessary
    ccf_domain_in_af_space = ccf_bins * mult * purity / (total_cn * purity + 2 * (1 - purity))
    ccf_dist = scipy.stats.beta.pdf(ccf_domain_in_af_space, alt_cnt + 1, ref_cnt + 1)
    return ccf_dist / sum(ccf_dist)

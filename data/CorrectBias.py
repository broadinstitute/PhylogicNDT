##WC_Correction

import random
import itertools
import numpy as np
from scipy.stats import binom


def est_ccf_dist(alt, ref, allele_cn, PURITY, other_cn, mult):
    ccf_dist = np.zeros(501)
    ccf_space = np.linspace(0, 1, 501)

    af_points = []

    for mult_1_bin_val_idx, mult_1_bin_val in enumerate(ccf_space):
        af = (mult * PURITY * mult_1_bin_val) / (
                (allele_cn * PURITY + other_cn * PURITY + 2 * (1 - PURITY)) + (
                PURITY * mult + (allele_cn - mult) * PURITY - allele_cn * PURITY) * (mult_1_bin_val))
        af_points.append(af)
        # ccf_dist[mult_1_bin_val_idx] = m1_draw

    ccf_dist[:] = binom._pmf(alt, alt + ref, np.array(af_points))

    ccf_dist[np.isnan(ccf_dist)] = 0.
    return ccf_dist / sum(ccf_dist)


def simulate_mutations(PURITY, NMUT, ccf, mutations):
    random.shuffle(mutations)
    muts_gen = itertools.cycle(mutations)

    clust_dist = np.zeros(501)

    mut_num = 0
    total_mut = 0
    while mut_num < NMUT and total_mut < 100 * NMUT:

        mut = muts_gen.next()
        if mut.type != "SNP":
            continue
        real_cov = mut.alt_cnt + mut.ref_cnt
        if real_cov == 0:
            continue
        a1, a2 = mut.local_cn_a1, mut.local_cn_a2
        if sum(np.isnan([a1, a2])) > 0:
            continue
        cn = a1 + a2

        if cn == 0:
            continue

        mult = np.random.choice([x for x in [a1, a2] if x > 0] + range(1, int(max([a1, a2])) + 1))

        af = (ccf * mult * PURITY) / (
                float(ccf) * mult * PURITY + ccf * (cn - mult) * PURITY + (1 - ccf) * (cn) * PURITY + 2 * (
                1.0 - PURITY))

        alt_count = np.random.binomial(real_cov, af)
        ref_count = int(real_cov - alt_count)

        total_mut += 1

        if alt_count >= 1:
            mut_num += 1
            ccf_dist = est_ccf_dist(alt_count, ref_count, a1, PURITY, a2, mult)
            clust_dist += np.log(ccf_dist)

    return clust_dist


def pre_compute_WCC(sample):
    print "pre-computing WCC"
    corr_y = []

    for obs_cluster_pos in list(np.logspace(0, 1, 25) / 10. - 0.1) + [1.]:

        err = 1
        cluster_pos = obs_cluster_pos
        while abs(err) > 0.005 and cluster_pos > 0:
            sim_res = np.average(
                [np.argmax(simulate_mutations(sample.purity, 500, cluster_pos, sample.concordant_variants)) / 500. for x
                 in range(5)])
            err = obs_cluster_pos - sim_res  # distance between observed cluster position, and the observed simulated cluster position.
            cluster_pos += err / 2.
            print cluster_pos,
        print "."
        corr_y.append(max(cluster_pos, 0))

    from scipy.interpolate import interp1d

    print "done pre-computing WCC!"

    return interp1d(list(np.logspace(0, 1, 25) / 10. - 0.1) + [1.], corr_y)

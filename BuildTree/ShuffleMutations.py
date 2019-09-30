import numpy as np
import random
import logging
from emd import emd

from scipy.misc import logsumexp as logsumexp_scipy


def logsum_of_marginals_per_sample(loghist):
    return np.apply_along_axis(lambda x: logsumexp_scipy(x), 1, np.array(loghist, dtype=np.float32))


def shuffling(clustering_results, sample_list):
    for cluster_idx, cluster in clustering_results.items():
        cluster.reset_counts()
    count_moved_mutations = 0
    mutations = sample_list[0].concordant_variants
    random.shuffle(mutations)
    n_clusters = len(clustering_results)
    n_samples = len(sample_list)
    for mut in mutations:
        loglik = np.ones((n_clusters, n_samples), dtype=np.float64) * -np.inf
        pre_cluster_assignment = mut.cluster_assignment
        mut_nd_hist = clustering_results[pre_cluster_assignment].mutations[mut]

        for cluster_idx, cluster in clustering_results.items():
            # Skip if the current point is the only thing in the cluster
            if cluster.cluster_size == 1 and mut.cluster_assignment == cluster_idx:
                logging.debug('Cluster {} has one mutation {}'.format(cluster.identifier, mut))
                continue

            if mut not in cluster.mutations:
                loglik[cluster_idx - 1] = logsum_of_marginals_per_sample(cluster.loghist + mut_nd_hist)
            else:
                cluster_loghist = cluster._normalize_loghist_with_prior(cluster.loghist - mut_nd_hist)
                loglik[cluster_idx - 1] = logsum_of_marginals_per_sample(cluster_loghist + mut_nd_hist)

        loglik = np.sum(loglik, axis=1)
        loglik = loglik - logsumexp_scipy(loglik)
        c_lik = np.exp(loglik - logsumexp_scipy(loglik))
        new_cluster_idx = np.nonzero(np.random.multinomial(1, c_lik) == 1)[0][0] + 1
        logging.debug('Mutation {} old cluster assignment {}, new cluster assignment {}'.format(mut.var_str,
                                                                                                mut.cluster_assignment,
                                                                                                new_cluster_idx))
        logging.debug('Cluster likelihoods {}'.format(c_lik))
        if new_cluster_idx != mut.cluster_assignment:
            count_moved_mutations += 1
            logging.debug('Mutation {} was assigned to new cluster {}'.format(mut.var_str, str(new_cluster_idx)))
            old_cluster = clustering_results[mut.cluster_assignment]
            new_cluster = clustering_results[new_cluster_idx]
            old_cluster.remove_mutation(mut, update_cluster_hist=True)
            new_cluster.add_mutation(mut, mut_nd_hist, update_cluster_hist=True, create_mut_nd_hist=False)
            mut.cluster_assignment = new_cluster_idx
    logging.debug(
        'Moved {} mutations out of {} mutations to new clusters'.format(count_moved_mutations, len(mutations)))
    for cluster_idx, cluster in clustering_results.items():
        logging.debug('Cluster {}, mutations added={}, removed={}'.format(cluster_idx,
                                                                          cluster._iter_count_added,
                                                                          cluster._iter_count_removed))


def emd_nd(u, v):
    """
    Computes Earth Mover's Distance in N-dimensions
    Uses https://github.com/garydoranjr/pyemd
    Need to convert probability distribution in non-log space
    """
    return emd(np.exp(u), np.exp(v))


def fix_cluster_lables(clustering_results):
    """ Gathers probability distributions for all clusters  """
    n_clusters = len(clustering_results)
    cluster_densities = []
    orig_dens_tp = []
    for cluster_idx, cluster in clustering_results.items():
        cluster_densities.append(cluster.loghist)
        orig_dens_tp.append(cluster.logprior)
    cluster_densities = np.asarray(cluster_densities, dtype=np.float32)
    orig_dens_tp = np.asarray(orig_dens_tp, dtype=np.float32)
    return get_labels_mapping(cluster_densities, orig_dens_tp, n_clusters)


def get_labels_mapping(cluster_densities, orig_dens_tp, n_clusters):
    """ For all pairwise combinations of clusters computes distance between
        between cluster prior and after mutation shuffling distributions """
    dist = []
    for c_n_o, cluster_old in enumerate(orig_dens_tp):
        if c_n_o == 0:
            continue
        for c_n_i, cluster_new in enumerate(cluster_densities):
            if c_n_i == 0:
                continue
            dist.append([emd_nd(cluster_old, cluster_new), c_n_o, c_n_i])
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
        new_densities.append(cluster_densities[mapping[c_n_o]])
    return mapping

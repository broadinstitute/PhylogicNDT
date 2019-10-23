# Import Statements

# Backend configuration necessary for execution in Docker. Need to import before other things.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import itertools
import random
from scipy.stats import binom
import networkx as nx
import numpy as np
import warnings

# Debug condition
DEBUG = False

# Global Constants: hg19 reference
hg19_names = ['chr13', 'chr12', 'chr11', 'chr10', 'chr17', 'chr16', 'chr15', 'chr14', 'chr19', 'chr18', 'chr22',
              'chr20', 'chr21', 'chr7', 'chr6', 'chr5', 'chr4', 'chr3', 'chr2', 'chr1', 'chr9', 'chr8']
hg19_lenghts = [115169878, 133851895, 135006516, 135534747, 81195210, 90354753, 102531392, 107349540, 59128983,
                78077248, 51304566, 63025520, 48129895, 159138663, 171115067, 180915260, 191154276, 198022430,
                243199373, 249250621, 141213431, 146364022]
hg19_start_end = [(1, 115169879), (1, 133851896), (1, 135006517), (1, 135534748), (1, 81195211), (1, 90354754),
                  (1, 102531393), (1, 107349541), (1, 59128984), (1, 78077249), (1, 51304567), (1, 63025521),
                  (1, 48129896), (1, 159138664), (1, 171115068), (1, 180915261), (1, 191154277), (1, 198022431),
                  (1, 243199374), (1, 249250622), (1, 141213432), (1, 146364023)]

# Randomly generated copy number segements for hg19
hg19_totalCN = [2, 3, 2, 4, 3, 4, 2, 2, 2, 4, 3, 2, 2, 4, 1, 4, 4, 4, 4, 2, 1, 2]
hg19_alCN = [(1, 1), (1, 2), (1, 1), (1, 3), (1, 2), (1, 3), (1, 1), (1, 1), (1, 1), (2, 2), (1, 2), (1, 1), (1, 1),
             (1, 3), (1, 0), (1, 3), (2, 2), (3, 1), (1, 3), (1, 1), (1, 0), (1, 1)]


# TODO: Add a growth simulation input (exponentially growing clones coeff)


###SUPPORTING FUNCTIONS###
def load_segfile(segfile):
    """
    Read a seg file with columns "ID, Chromosome, Start_Position, End_Position, Major_CN, Minor_CN", where the
    major and minor CN are the two possible allelic copy numbers of the segment. Returns ordered segment values.
    Fields are separated by tabs (tsv).


    Args:
        segfile: path of tsv seg file to load

    Returns:
        A dictionary containing the segments names, start and end positions, lengths, allelic CNs and
        total CN in lists for each category. Start and end positions and allelic copy numbers are reported in lists
        of tuples.

    """

    seg_file_raw = open(segfile)
    header = tsv_line(seg_file_raw.next())

    if header[1].lower() != 'chromosome' or header[2].lower() != 'start_position' or header[
        3].lower() != 'end_position':
        print "HEADER:"
        print header
        raise HeaderError(
            'The seg file columns should follow the structure ID, Chromosome, Start_Position, End_Position'
            '. Current header incorrect.')

    # List of strings
    segm_names = []
    # List of int tuples
    segm_start_end = []
    # List of ints
    segm_length = []
    # List of int tuples
    segm_Al_CN = []
    # List of ints
    segm_Tot_CN = []

    for line in seg_file_raw:
        data = tsv_line(line)

        # Check subclonal CN
        try:
            tot_CN = int(data[4]) + int(data[5])
        except ValueError:
            warnings.warn('The segment %s, %s, %s contains a subclonal CN profile, which is currently ignored. '
                          'Subclonal CN events will be included in the future.' % (data[1], data[2], data[3]),
                          FutureWarning)
            continue

        # Check for homdel segment, ignore if present.
        if tot_CN == 0:
            warnings.warn('The segment %s, %s, %s has copy number zero, and therefore will be '
                          'ignored.' % (data[1], data[2], data[3]))
            continue

        segm_Al_CN.append((int(data[4]), int(data[5])))
        segm_Tot_CN.append(tot_CN)
        segm_names.append(data[1])
        segm_start_end.append((int(data[2]), int(data[3])))
        segm_length.append(int(data[3]) - int(data[2]))

    segments_data = {}
    segments_data['names'] = segm_names
    segments_data['lengths'] = segm_length
    segments_data['start_end'] = segm_start_end
    segments_data['al_CN'] = segm_Al_CN
    segments_data['tot_CN'] = segm_Tot_CN

    return segments_data


def tsv_line(line):
    """
    Apply the standard string transformation for a line in a tsv file: strip the end
    of line character, and split by tab.

    Args:
        line: str

    Returns:
        List of the elements in the string that were separated by tab.

    """
    output = line.strip('\n').split('\t')
    return output


class HeaderError(ValueError):
    pass


class InputError(ValueError):
    pass


def make_clones_plot(true_cluster_ccfs, true_mutnumber, purity, fig=None, ax=None):
    """
    Make a line trajectory plot given the modeled cluster ccfs (matrix with row as single clone ccfs, the number of
    mutations in each cluster (dict with cluster ids as keys, values as mut numbers) and the sample purities (list)

    Returns: fig (figure object), ax (axis handle)

    """

    # Find the number of samples and the number of clusters
    num_samples = len(true_cluster_ccfs[0])
    num_clusters = len(true_mutnumber)

    if fig is None and ax is None:
        fig = plt.figure(figsize=(min(2 + 2 * num_samples, 25), 8))
    if ax is None:
        ax = plt.gca()

    # Plot the real values
    for idx in range(1, num_clusters + 1):
        points = true_cluster_ccfs[idx - 1]
        ax.plot(points, marker='o', linestyle='--', markersize=8,
                label=('True Cluster #' + str(idx) + ' (' + str(true_mutnumber[idx]) + ')'))

    # Add Legend and Levels and title
    X = range(num_samples)
    ax.set_xticks(X)

    ax.set_title('Simulated Samples Clone Trajectories', fontsize=20)

    labels = ['Sample ' + str(x) + '\n purity = ' + str(purity[x - 1]) for x in range(1, num_samples + 1)]
    if num_samples > 10:
        ax.set_xticklabels(labels, fontsize=8)
    else:
        ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('CCF', fontsize=15)
    ax.legend(fontsize=12)

    return fig, ax


def rand_betabin(alpha, beta, nbin, n=1):
    """
    Random numbers drawn from a betabinomial distribution. Returns a numpy array of draws of size n.
    """

    beta_p = np.random.beta(alpha, beta, n)
    draw = np.random.binomial(nbin, beta_p, size=n)
    return draw



###TRUTH GENERATION###
# Create a truth set of clusters and CCFs
def gen_tree(min_nodes):
    good_tree = False
    while not good_tree:
        tree = nx.generators.directed.gn_graph(np.random.randint(min_nodes, min_nodes + 1)).reverse()
        if max([len(get_siblings(node, tree)) for node in tree.nodes()]) < 4: good_tree = True
    return tree


def nx_walk(node, tree):
    """ iterate tree in pre-order depth-first search order """
    yield node
    for child in sorted(tree.successors(node), key=lambda x: np.random.random()):
        for n in nx_walk(child, tree):
            yield n


def get_siblings(node, tree):
    try:
        return tree.successors(tree.predecessors(node)[0])
    except:
        return []


def gen_sample_from_tree_all(tree):
    leaves = [node for node in tree.nodes() if len(tree.successors(node)) == 0]
    leaves_to_include = leaves
    non_zero_clusters = set(sum([list(nx_walk(x, tree.reverse())) for x in leaves_to_include], []))

    clusters = [0.] * len(tree.nodes())

    while min(np.diff(sorted(clusters))) < 0.001:  # run once - disabled this check
        clusters = [0.] * len(tree.nodes())
        for node in nx_walk(0, tree):
            if node in non_zero_clusters:
                if len(tree.predecessors(node)) == 0:  # clonal case
                    clusters[node] = 1.
                    continue

                clusters[node] = (1 - random.random()) * (
                            clusters[tree.predecessors(node)[0]] - sum([clusters[x] for x in get_siblings(node, tree)]))
            else:
                continue
            # if len(tree.predecessors(node)) > 0:

    return [round(x, 2) for x in clusters]


###MUTATION SAMPLING###
# Generate a set of mutations from the truth set and related functions
def get_ccf_no_adj(mult, cn, alt, ref, PURITY, grid_size=101):
    """
    Calculate the ccf distribution for a mutation given it's alt and ref count,
    purity, cn and multiplicity.

    Uses a binomial to compute the distirbution, which as of this writting, does
    not make sense.

    Parameters
    ----------
    mult : int
        multiplicity of alternate allele
    cn : int
        total copy number
    alt : type
        number of alternate reads
    ref : type
        number of reference reads
    PURITY : type
        sample purity
    grid_size : type
        number of bins in which to divide the ccf histogram.

    Returns
    -------
    numpy.array
        Numpy array of size grid_size with the normalized ccf distribution.

    """

    ccf_space = np.linspace(0, 1, grid_size)
    ccf_dist = np.zeros(grid_size)

    for mult_1_bin_val_idx, mult_1_bin_val in enumerate(ccf_space):
        x = mult_1_bin_val * mult * PURITY / (
                    float(mult_1_bin_val) * mult * PURITY + mult_1_bin_val * (cn - mult) * PURITY + (
                        1 - mult_1_bin_val) * (cn) * PURITY + 2 * (1.0 - PURITY))
        m1_draw = binom.pmf(alt, alt + ref, x)
        ccf_dist[mult_1_bin_val_idx] = m1_draw

    ccf_dist[np.isnan(ccf_dist)] = 0.
    return ccf_dist / sum(ccf_dist)


def make_sim_ccfs_ND(clones_ar_nd=[[1.0, .8, .4, .25, .12], [1.0, .6, .75, .1, .55], [1.0, .4, .8, .1, .65],
                                   [1.0, .2, 1.0, .1, .75]], mult_ar=[1, 2, 2, 2, 2, 2, 3, 4, 4], NMUT=200,
                     PURITY=[0.7, 0.7, 0.7, 0.7, 0.7], seed=157, clust_props=None, alpha=[2, 2, 2, 2, 2],
                     beta=[18, 18, 18, 18, 18], nbin=[1000, 1000, 1000, 1000, 1000], coverage_gen=None,
                     segment_dict=None):

    # First, select a contig and position for each mutation
    contig_names = segment_dict['names']
    contig_length = segment_dict['lengths']
    contig_start_end = segment_dict['start_end']
    contig_totalCN = segment_dict['tot_CN']
    contig_AlCN = segment_dict['al_CN']

    # Normalize vector of chromosome lenghts
    contig_length = np.array(contig_length)
    contig_prob = contig_length / float(contig_length.sum())

    mutation_contigs = []
    mutation_pos = []
    mutation_total_cn = []
    mutation_multiplicity = []

    print "Sampling mutations"
    for i in range(NMUT):
        # Sample the contig
        contig_idx = np.argmax(np.random.multinomial(1, contig_prob))
        mutation_contigs.append(contig_names[contig_idx])

        # Sample the position
        contig_start, contig_end = contig_start_end[contig_idx]
        mutation_pos.append(random.randrange(contig_start, contig_end))

        # Get the copy number
        mutation_total_cn.append(contig_totalCN[contig_idx])
        al_1, al_2 = contig_AlCN[contig_idx]
        al_CN = [al_1, al_2]

        # Choose a random allelic copy number, and make sure it's not zero.
        # Since no 0 copy number segments are allowed, there is at least one al_CN
        # that is no zero.
        random.shuffle(al_CN)
        new_alCN = al_CN[0]
        if new_alCN == 0:
            new_alCN = al_CN[1]

        # Sample the multiplicity
        mutation_multiplicity.append(random.choice(range(1, new_alCN + 1) + [1]))  # more mult 1

    ND_ccfs = []
    ND_truths = {}
    print 'Preparing sample 0'
    ccf_modeled_A, ccf_real_A, ccf_idx, more3_alt, real_abscn, real_multp, real_cov, alt_ref = make_sim_ccfs_1D(
        clones_ar=clones_ar_nd[0], mult_ar=mult_ar, NMUT=NMUT, real_clust_order=None, PURITY=PURITY[0], seed=seed,
        clust_props=clust_props, alpha=alpha[0], beta=beta[0], nbin=nbin[0], coverage_gen=coverage_gen,
        real_abscn=mutation_total_cn, real_multp=mutation_multiplicity)

    ND_ccfs.append((ccf_modeled_A, ccf_real_A, ccf_idx))
    ND_truths[0] = {'real_abscn': real_abscn, 'real_multp': real_multp, 'real_cov': real_cov, 'alt_ref': alt_ref}

    mut_order = ccf_idx
    passing_muts = more3_alt

    if DEBUG:
        print "Mut Order"
        print mut_order

    for i in range(1, len(clones_ar_nd)):
        print 'Preparing sample ' + str(i)
        ccf_modeled, ccf_real, ccf_idx, more3_alt, real_abscn, real_multp, real_cov, alt_ref = make_sim_ccfs_1D(
            clones_ar=clones_ar_nd[i], mult_ar=mult_ar, NMUT=NMUT, alpha=alpha[i], beta=beta[i], nbin=nbin[i],
            coverage_gen=coverage_gen, real_clust_order=mut_order, PURITY=PURITY[i], seed=seed,
            real_abscn=mutation_total_cn, real_multp=mutation_multiplicity)
        ND_ccfs.append((ccf_modeled, ccf_real))
        ND_truths[i] = {'real_abscn': real_abscn, 'real_multp': real_multp, 'real_cov': real_cov, 'alt_ref': alt_ref}
        passing_muts = more3_alt | passing_muts

    return ND_ccfs, passing_muts, ND_truths, mutation_contigs, mutation_pos


# TODO: Optimize this function to avoid loops through all the mutations: Make a single function per mut.
def make_sim_ccfs_1D(clones_ar=[1.0, .8, .4, .25, .12], mult_ar=[1, 2, 2, 2, 2, 2, 3, 4, 5],
                     NMUT=200, real_clust_order=None, PURITY=0.7, seed=157, clust_props=None,
                     alpha=2, beta=18, nbin=1000, coverage_gen=None, real_abscn=None, real_multp=None):
    # define mutations and their real CCFs
    real_ccfs = []
    ccf_idx = []

    if not real_clust_order:  # if not pre-defined
        for i in xrange(NMUT):
            m_c_idx = np.argmax(np.random.multinomial(1, clust_props))
            ccf_idx.append(m_c_idx)
            if clones_ar[m_c_idx] >= 0:
                real_ccfs.append(clones_ar[m_c_idx])
            else:
                real_ccfs.append(random.random())  # if artifact
    else:
        for i in xrange(NMUT):
            if clones_ar[real_clust_order[i]] >= 0:
                real_ccfs.append(
                    clones_ar[real_clust_order[i]])  # if pre assigned cluster membership , mostly for ND simulation
            else:
                real_ccfs.append(random.random())  # if artifact

    # assign abs copy numbers

    # Leaving old method of cn assignment for reproducibility/backwards comp.
    if real_abscn == None:
        real_abscn = []
        for i in xrange(NMUT):
            # Make sure that the copy number is not zero
            new_cn = 0
            while new_cn == 0:
                new_cn = random.choice(mult_ar)
            real_abscn.append(new_cn)
    else:
        if DEBUG:
            print 'Using ND provided CN'

    # assign multipl-s
    if real_multp == None:
        real_multp = []
        for i in real_abscn:
            real_multp.append(random.choice(range(1, i + 1) + [1]))  # more mult 1
    else:
        if DEBUG:
            print 'Using provided multiplicity'

    # calc real AFs
    real_AFs = []
    for cn, mult, ccf in zip(real_abscn, real_multp, real_ccfs):
        af = ccf * mult * PURITY / (
                    float(ccf) * mult * PURITY + ccf * (cn - mult) * PURITY + (1 - ccf) * (cn) * PURITY + 2 * (
                        1.0 - PURITY))
        real_AFs.append(af)

    # get coverage
    real_cov = []
    for total_cn in real_abscn:

        # If a file was passed, just sample from the file.
        if coverage_gen is not None:
            real_cov.append(coverage_gen.next())

        # If a file was not passed, use the beta binomial.
        elif total_cn == 2:
            real_cov.append(rand_betabin(alpha, beta, nbin, n=1)[0])

        else:
            # fix alpha and n, recalculate beta to match the new expected mean coverage given the copy number
            scaled_mean_cov = 100 * (1 - PURITY) + 100 * PURITY * total_cn / 2.
            new_beta = alpha * (nbin - scaled_mean_cov) / scaled_mean_cov
            if new_beta <= 0:
                print new_beta, total_cn
                print 'NEW BETA BELOW 0, renormalizing to original beta'
                new_beta = beta
            real_cov.append(rand_betabin(alpha, new_beta, nbin, n=1)[0])

    # Boolean array to keep track of mutations that have 3 or more alt count.
    more3_alt = []

    # given real AF generate alt,ref count with given coverage
    alt_ref = []
    for n, af in zip(real_cov, real_AFs):
        p = af
        alt = np.random.binomial(n, p)
        if alt >= 3:
            more3_alt.append(True)
        else:
            more3_alt.append(False)
        alt_ref.append((alt, n - alt))

    more3_alt = np.array(more3_alt)

    ccf_hist = []
    for cn, mult, ccf, (alt, ref) in zip(real_abscn, real_multp, real_ccfs, alt_ref):
        ccf_hist.append(get_ccf_no_adj(mult, cn, alt, ref, PURITY))

    return ccf_hist, real_ccfs, ccf_idx, more3_alt, real_abscn, real_multp, real_cov, alt_ref


###MAIN SIMULATION FUNCTION###
def run_simulations(args):
    ### CLUSTER CCF GENERATION ###
    if args.clust_file == None:
        print 'Generating a random tree'
        # Generate a Random Tree
        good_sim = 0
        tree = gen_tree(args.min_nodes)

        # Code to make sure the simulation is good.
        while good_sim == 0:

            # Randomly regenerate the tree to prevent it from getting stuck in a bad tree if we have been running
            # this loop a bunch of times.
            if random.random() < 0.01:
                tree = gen_tree(args.min_nodes)

            # For as many samples as necessary, sample the ccfs plus a cluster of artifacts. If only one sample, it samples 2?
            clusters = [gen_sample_from_tree_all(tree) + [-1] for x in range(max(args.nsamp, 2))]

            # Loops to check that clusters are not identical.
            for c_idx, clust in enumerate(np.array(clusters).T):
                if min(clust) < 0: continue  # skip artifacts
                good_clust = 0
                for s_c_idx, s_clust in enumerate(np.array(clusters).T):
                    if s_c_idx == c_idx: continue
                    if min(s_clust) < 0: continue  # skip artifacts

                    if max(np.abs(clust - s_clust)) < 0.3: break
                else:
                    good_clust = 1

                # If it finds a bad cluster, regenerate the ccfs/resample tree.
                if good_clust == 0: break

            # This executed at the end of the for loop run, once we have checked that all the clusters and made sure they are ok.
            else:
                good_sim = 1

        print 'Tree Edges:'
        print tree.edges()

    # If a cluster file is specified, read it.
    else:
        print 'Reading clusters from file ' + args.clust_file
        clusters = []
        file = open(args.clust_file, 'r')
        for line in file:
            new_timepoint = line.strip('\n').split('\t')
            new_timepoint = [float(x) for x in new_timepoint]
            clusters.append(new_timepoint)

        # Change the number of samples to the ones in the file.
        args.nsamp = len(clusters)

        # Change the number of clusters to the ones in the file
        args.min_nodes = len(clusters[0]) - 1
        if args.min_nodes < 2:
            raise InputError("The cluster file needs to specify at least one subclonal clone")

    print 'Clusters:'
    print clusters

    ### SET PURITY AND COVERAGE INPUTS ###
    # First declare the given values, then replace if a file is provided.
    alpha = [args.alpha] * args.nsamp
    beta = [args.beta] * args.nsamp
    nbin = [args.nbin] * args.nsamp
    purity = [args.purity] * args.nsamp

    # Read the values from the file if provided
    if args.purity_file is not None:
        print 'Using provided purity file'
        file = open(args.purity_file, 'r')
        alpha = []
        beta = []
        nbin = []
        purity = []
        for line in file:
            new_sample = line.strip('\n').split('\t')
            if len(new_sample) > 1:
                purity.append(float(new_sample[0]))
                alpha.append(float(new_sample[1]))
                beta.append(float(new_sample[2]))
                nbin.append(int(new_sample[3]))
            else:
                purity.append(float(new_sample[0]))
                alpha.append(args.alpha)
                beta.append(args.beta)
                nbin.append(args.nbin)

        if len(purity) < args.nsamp:
            raise InputError(
                "Purity file contains less purity values (%d) than samples to simulate (%d)" % (len(purity), args.nsamp))


    print "Purity:"
    print purity

    #Generate the Coveage
    if args.cov_file is not None:
        print 'Using provided coverage file to sample coverage.'
        coverage_samp = [int(x.strip("\n")) for x in open(args.cov_file).readlines() if int(x.strip("\n")) > 0]
        random.shuffle(coverage_samp)
        coverage_samp = itertools.cycle(coverage_samp)

    else:
        print 'Using betabinomial coverage distribution with alpha ' + str(alpha) + ' beta ' + str(
            beta) + ' and n ' + str(nbin)
        print 'Binomial mean will be modified in cases where copy number is different from 2'
        coverage_samp = None

    # Number of values to calculate for the ccf. Why is this not a parametere that's passed in?
    global grid_size
    grid_size = 101

    ### SET CLUSTER PROPORTIONS OF MUTATIONS ###
    # Cluster proportions, that is, the percentage of mutations in each cluster.
    # This proportion is assigned at random and then renormalized. args.artifacts represents
    # the proportion of mutations that are artifacts.
    clust_props=[random.random()+0.1 for x in sorted(clusters)[0]]
    clust_props=[x/sum(clust_props[:-1])*(1-args.artifacts) for x in clust_props[:-1]]+[args.artifacts]
    print "Cluster Proportions:"
    print clust_props

    ### PREPARE CN INPUT ###
    # TODO: Remove cn_dist once it's deprecated
    # Read the cn distribution file. If no segments file provided, use the standard hg19 values.
    if args.cn_dist == None:
        cn_dist = [1, 2, 2, 2, 2, 2, 3, 4, 4]
        segment_data = {}
        warnings.warn('No segment file provided, using hg19 whole genome segments and example CN distribution. Randomly'
                      ' generated copy number distributions will be implemented in the future.', FutureWarning)
        segment_data['names'] = hg19_names
        segment_data['lengths'] = hg19_lenghts
        segment_data['start_end'] = hg19_start_end
        segment_data['tot_CN'] = hg19_totalCN
        segment_data['al_CN'] = hg19_alCN

    else:
        print 'Using provided segments file to sample absolute copy number.'
        segment_data = load_segfile(args.cn_dist)
        cn_dist = segment_data['tot_CN']

    print "Total Copy Number distribution:"
    print cn_dist

    # TODO: Add a maximum mutation number

    # TODO: In the future, output a segment file as well.

    if DEBUG:
        print "Coverage method:"
        print coverage_samp

    ### RUN SIM ###
    a, passing_muts, sym_truths, mutation_contigs, mutation_pos = make_sim_ccfs_ND(clones_ar_nd=clusters,
                                                                                   clust_props=clust_props,
                                                                                   NMUT=args.nmuts, PURITY=purity,
                                                                                   mult_ar=cn_dist, alpha=alpha,
                                                                                   beta=beta, nbin=nbin,
                                                                                   coverage_gen=coverage_samp,
                                                                                   segment_dict=segment_data)

    # Creating sif file for this simulations
    header = ['sample_id', 'maf_fn', 'seg_fn', 'purity', 'timepoint']
    import inspect, os
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    with open(args.indiv_id + '.sif', 'w') as writer:
        writer.write('\t'.join(header) + '\n')
        for j in range(len(a)):
            sample_name = args.indiv_id + "_" + str(j)
            line = [sample_name, path + '/' + sample_name + ".txt", '', str(args.purity), str(j)]
            writer.write('\t'.join(line) + '\n')

    ### OUTPUT ###
    for j in range(len(a)):
        with open(args.indiv_id + "_" + str(j) + ".txt", "w") as outccf:
            outccf.write("\t".join(
                ["Hugo_Symbol", "Chromosome", "Start_position", "Reference_Allele", "Tumor_Seq_Allele2", "t_ref_count",
                 "t_alt_count"]) + "\t")
            outccf.write("\t".join(['ccf_' + str('%.2f' % num) for num in np.linspace(0, 1, grid_size)]) + "\n")

            print "Writing sample " + str(j)

            sample_truth = sym_truths[j]
            for i, line in enumerate(a[j][0]):

                alt, ref = sample_truth['alt_ref'][i]
                cov = sample_truth['real_cov'][i]
                if DEBUG:
                    print alt + ref, cov
                    print alt + ref == cov
                    print alt, ref
                    print a[j][1][i]
                # Check that the mutation passes 3 alt reads the condition
                if passing_muts[i]:
                    line[line < 0] = 0
                    outccf.write("\t".join(
                        ["Unknown", str(mutation_contigs[i]), str(mutation_pos[i]), "A", "T", str(ref),
                         str(alt)]) + "\t")
                    outccf.write("\t".join(map(str,line))+"\n")

    with open(args.indiv_id + ".truth.txt", "w") as outccf:
        outccf.write("\t".join(
            ["mut_idx\tcontig\tposition\tcluster_id\ttotalCN\tmultiplicity"] + ["ccf_" + str(x) for x in
                                                                                range(args.nsamp)]) + "\n")
        for i, line in enumerate(a[0][1]):
            # Check that the mutation passes 3 alt reads the condition
            if passing_muts[i]:
                outccf.write("\t".join([str(i + 1), str(mutation_contigs[i]), str(mutation_pos[i]), str(a[0][2][i]),
                                        str(sym_truths[0]['real_abscn'][i]), str(sym_truths[0]['real_multp'][i])]))
                for j in range(len(a)):
                    outccf.write("\t" + str(a[j][1][i]))
                outccf.write("\n")

    print 'Number of Mutations that fullfill 3 alt read condition:'
    print np.sum(passing_muts == True)

    # Find the number of mutations per cluster. The extra cluster is artifacts.
    num_mut = {}
    for i in range(1, args.min_nodes + 2):
        num_mut[i] = 0

    counter = 0
    for i in a[0][2]:
        if passing_muts[counter]:
            num_mut[i + 1] = num_mut[i + 1] + 1
        counter = counter + 1

    # Extract the artifacts
    artifact_nmuts = num_mut.pop(args.min_nodes + 1, 0)

    print "Number of valid artifact mutations"
    print artifact_nmuts

    write_out_list = []

    with open(args.indiv_id + ".simulation_paramters.tsv", "w") as p_file:
        write_out_list.append("id\t" + str(args.indiv_id) + "\n")
        write_out_list.append("purity\t" + str(purity) + "\n")
        write_out_list.append("n_muts\t" + str(args.nmuts) + "\n")
        write_out_list.append("n_samps\t" + str(args.nsamp) + "\n")
        write_out_list.append("num_clust\t" + str(args.min_nodes) + "\n")
        write_out_list.append("clusters\t" + str(clusters) + "\n")
        write_out_list.append("prop\t" + str(clust_props) + "\n")
        write_out_list.append("alpha\t" + str(alpha) + "\n")
        write_out_list.append("beta\t" + str(beta) + "\n")
        write_out_list.append("nbin\t" + str(nbin) + "\n")
        write_out_list.append("frac_artifacts\t" + str(args.artifacts) + "\n")
        write_out_list.append("cn_dist_unused\t" + str(cn_dist) + "\n")
        write_out_list.append("num_valid_muts\t" + str(np.sum(passing_muts == True)) + "\n")
        write_out_list.append("num_valid_artifactmuts\t" + str(artifact_nmuts) + "\n")

        if args.clust_file == None:
            write_out_list.append("nodes\t" + str(len(tree.nodes())) + "\n")
            write_out_list.append("edges\t" + str(tree.edges()) + "\n")
            write_out_list.append("clust_file\tNA\n")
        else:
            write_out_list.append("nodes\tNA\n")
            write_out_list.append("edges\tNA\n")
            write_out_list.append("clust_file\t" + str(args.clust_file) + "\n")

        if args.cov_file is not None:
            write_out_list.append("cov_file\t" + str(args.cov_file) + "\n")
        else:
            write_out_list.append("cov_file\tNA\n")

        if args.purity_file is not None:
            write_out_list.append("purity_file\t" + str(args.purity_file) + "\n")
        else:
            write_out_list.append("purity_file\tNA\n")

        write_out_string = ''.join(write_out_list)
        p_file.write(write_out_string)

    # Reorder the cluster ccfs to input into the plotting function
    temp_clust = np.matrix(clusters)
    true_clust_CCF = temp_clust.transpose().tolist()

    pdf_fig, (ax1, ax2) = plt.subplots(2, figsize=(25, 16))
    pdf_fig.suptitle(args.indiv_id + ' Simulation Results', fontsize=20)

    write_out_plot = write_out_string.replace('\t', ' ')
    ax1.text(0.05, 0.95, write_out_plot, fontsize=15, verticalalignment='top', transform=ax1.transAxes)
    ax1.axis('off')

    none_fig, ax2 = make_clones_plot(true_clust_CCF, num_mut, purity, ax=ax2)
    pdf_fig.savefig(args.indiv_id + '_report.pdf', bbox_inches='tight')


    out_fig, out_ax = make_clones_plot(true_clust_CCF, num_mut, purity)
    out_fig.savefig(args.indiv_id + '_plot.png', bbox_inches='tight')

    if DEBUG:
        print "CLUSTER DEBUGGING:"
        print "clusters"
        print clusters
        print "temp_clust"
        print temp_clust
        print "true_clust_CCF"
        print true_clust_CCF

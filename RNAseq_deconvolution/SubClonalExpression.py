import os
import logging
import numpy as np
from scipy.optimize import minimize
from BuildTree.ClusterObject import Cluster

# from RNASample import RNASample
# TODO: blacklist RPS
# TODO: exclude genes above 10
# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rna_expression_log.log')
print filename
logging.basicConfig(filename=filename,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=getattr(logging, "DEBUG"))


# Main run method
def run_tool(indiv_id, sif_file, mutation_ccf_file, cluster_ccf_file, abundance_file):
    # logging.debug('Arguments {}'.format(args))

    import data.Patient as Patient
    # from .CellPopulationEngine import CellPopulationEngine
    # from .BuildTreeEngine import BuildTreeEngine

    # init a Patient
    # patient_data = Patient.Patient(indiv_name=args.indiv_id, driver_genes_file=args.driver_genes_file)
    patient_data = Patient.Patient(indiv_name=indiv_id)

    # SubClonalExpression

    # Patient load cluster and mut ccf files
    # parse_sif_file(args.sif, args.mutation_ccf_file, patient_data)
    parse_sif_file(sif_file, mutation_ccf_file, patient_data)
    load_clustering_results(cluster_ccf_file, patient_data)
    patient_data.preprocess_rna_samples()
    cell_pop_abundances = load_abundance(abundance_file, patient_data)
    tpm_values = get_all_tpm_values(patient_data)
    solution = compute_solution(tpm_values.T, cell_pop_abundances)
    genes = patient_data.concordant_genes
    values = set()
    data = []
    for idx, gene_id in enumerate(genes):
        # values.add((sum(solution) * 100, e))

        for j in range(len(solution)):
            data.append(dict(
                gene_name=gene_id,
                cluster=j,
                expression=solution[j]
            ))
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv('test_solution.tsv', sep='\t', index=False)
    """
    
    patient_data.preprocess_samples()
    # Building Phylogenetic Tree
    bt_engine = BuildTreeEngine(patient_data)
    bt_engine.build_tree(n_iter=args.n_iter)
    # Assign Top tree to Patient
    patient_data.TopTree = bt_engine.top_tree
    patient_data.TreeEnsemble = bt_engine.trees

    # Computing Cell Population
    cp_engine = CellPopulationEngine(patient_data)
    constrained_ccf = cp_engine.compute_constrained_ccf(n_iter=args.n_iter)
    cell_ancestry = bt_engine.get_cell_ancestry()
    cell_abundance = cp_engine.get_cell_abundance(constrained_ccf)
    # Output and visualization
    import output.PhylogicOutput
    phylogicoutput = output.PhylogicOutput.PhylogicOutput()
    phylogicoutput.write_tree_tsv(bt_engine.mcmc_trace, bt_engine.trees_ll, args.indiv_id)
    phylogicoutput.write_constrained_ccf_tsv(constrained_ccf, cell_ancestry, args.indiv_id)
    phylogicoutput.write_cell_abundances_tsv(cell_abundance, cell_ancestry, args.indiv_id)
    phylogicoutput.generate_html_from_tree(args.mutation_ccf_file, args.cluster_ccf_file,
                                           args.indiv_id + '_build_tree_posteriors.tsv',
                                           args.indiv_id + '_constrained_ccf.tsv',
                                           sif=args.sif,
                                           drivers=patient_data.driver_genes,
                                           treatment_file=args.treatment_data,
                                           tumor_sizes_file=args.tumor_size, )
    # cnv_file=args.indiv_id + '.cnvs.txt')
    """


def parse_sif_file(sif_file, mutation_ccf_file, patient_data):
    with open(sif_file, 'r') as reader:
        for line in reader:
            if not line.strip() == "":
                # for now, assume input file order is of the type sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint
                values = line.strip('\n').split('\t')
                if line.startswith('sample_id'):
                    header = {x: i for i, x in enumerate(values)}
                else:
                    sample_id = values[header['sample_id']]
                    # TODO: incorporate copy number events in RNA deconvolution
                    rna_seq_fn = values[header['tpm_fn']]
                    purity = float(values[header['purity']])
                    timepoint = float(values[header['timepoint']])
                    logging.debug("Adding RNA sample {}".format(sample_id))
                    patient_data.addRNAsample(rna_seq_fn, sample_id, timepoint=timepoint, purity=purity)


def load_abundance(abundance_file, patient_data):

    n_samples = len(patient_data.rna_sample_list)
    # Add 1 to represent Normal cell population
    n_clusters = len(patient_data.ClusteringResults) + 1
    sample_names = [sample.sample_name for sample in patient_data.rna_sample_list]
    samples_purities = [sample.purity for sample in patient_data.rna_sample_list]
    abundances = np.zeros((n_samples, n_clusters))
    with open(abundance_file, 'r') as reader:
        for line in reader:
            values = line.strip('\n').split('\t')
            if line.startswith('Patient_ID'):
                header = {v: k for k, v in enumerate(values)}
            else:
                cluster = int(values[header['Cell_population']].split('_')[-1].strip('CL')) - 1
                sample_idx = sample_names.index(values[header['Sample_ID']])
                cluster_abundance = float(values[header['Cell_abundance']]) * samples_purities[sample_idx]
                abundances[sample_idx, cluster] = cluster_abundance / 100.
    for sample_idx, purity in enumerate(samples_purities):
        abundances[sample_idx, n_clusters - 1] = 1.0 - purity
    return abundances


def load_abundance(abundance_file, patient_data):
    n_samples = len(patient_data.rna_sample_list)
    # Add 1 to represent Normal cell population
    n_clusters = len(patient_data.ClusteringResults) + 1
    sample_names = [sample.sample_name for sample in patient_data.rna_sample_list]
    samples_purities = [sample.purity for sample in patient_data.rna_sample_list]
    abundances = np.zeros((n_samples, n_clusters))
    with open(abundance_file, 'r') as reader:
        for line in reader:
            values = line.strip('\n').split('\t')
            if line.startswith('Patient_ID'):
                header = {v: k for k, v in enumerate(values)}
            else:
                cluster = int(values[header['Cell_population']].split('_')[-1].strip('CL')) - 1
                sample_idx = sample_names.index(values[header['Sample_ID']])
                cluster_abundance = float(values[header['Cell_abundance']]) * samples_purities[sample_idx]
                abundances[sample_idx, cluster] = cluster_abundance / 100.
    for sample_idx, purity in enumerate(samples_purities):
        abundances[sample_idx, n_clusters - 1] = 1.0 - purity
    return abundances


def get_all_tpm_values(patient_data):
    genes = patient_data.concordant_genes
    n_samples = len(patient_data.rna_sample_list)
    tpm_values = np.zeros((n_samples, len(genes)))
    for gene_idx, gene_id in enumerate(genes):
        for sample_idx, sample in enumerate(patient_data.rna_sample_list):
            tpm_values[sample_idx, gene_idx] = sample.get_tpm_by_gene_id(gene_id)
    return tpm_values


def compute_solution(exp_values, abundances, method='L-BFGS-B'):
    n = abundances.shape[1]
    fun = lambda x: matrix_diff(np.dot(abundances, x), exp_values)
    sol = minimize(fun, np.ones(n), method=method, bounds=[(0.01, None) for x in xrange(n)])
    return sol.x


def matrix_diff(X, Y):
    return np.linalg.norm(X - Y)


def load_clustering_results(cluster_info_file, patient_data):
    clustering_results = {}
    ccf_headers = ['postDP_ccf_' + str(i / 100.0) for i in range(0, 101, 1)]
    sample_names = [sample.sample_name for sample in patient_data.rna_sample_list]
    with open(cluster_info_file, 'r') as reader:
        for line in reader:
            values = line.strip('\n').split('\t')
            if line.startswith('Patient_ID'):
                header = dict((item, idx) for idx, item in enumerate(values))
            else:
                sample_id = values[header['Sample_ID']]
                cluster_id = int(values[header['Cluster_ID']])
                ccf = [float(values[header[i]]) for i in ccf_headers]
                if cluster_id not in clustering_results:
                    new_cluster = Cluster(cluster_id, sample_names)
                    clustering_results[cluster_id] = new_cluster
                    logging.debug('Added cluster {} '.format(cluster_id))
                clustering_results[cluster_id].add_sample_density(sample_id, ccf)
    for cluster_id, cluster in clustering_results.items():
        cluster.set_blacklist_status()
        clustering_results[cluster_id] = cluster

    # Create for each cluster dictionary of mutations (key - mut_var_str and value- nd_histogram in log space)
    mutations_nd_hist = {}
    for sample in patient_data.sample_list:
        for mutation in sample.mutations:
            if mutation not in mutations_nd_hist:
                mutations_nd_hist[mutation] = []
            mutations_nd_hist[mutation].append(mutation.ccf_1d)
    for mutation, mutation_nd_hist in mutations_nd_hist.items():
        clustering_results[mutation.cluster_assignment].add_mutation(mutation, mutation_nd_hist)
    patient_data.ClusteringResults = clustering_results


def main():
    indiv = 'CLL-CRC-0001'
    sif_file = '/Users/lelagina/PycharmProjects/PhylogicNDT-BuildTree_rework/kipps/CRC-0001_RNA.sif'
    abundances_file = '/Users/lelagina/PycharmProjects/PhylogicNDT-BuildTree_rework/kipps/CRC-0001_cell_population_mcmc_trace.tsv'
    # abundances_file = '/Users/lelagina/PycharmProjects/PhylogicNDT-BuildTree_rework/kipps/CRC-0001_cell_population_abundances.tsv'
    cluster_ccf_file = '/Users/lelagina/PycharmProjects/PhylogicNDT-BuildTree_rework/kipps/CLL-CRC-0001.cluster_ccfs.txt'
    constrained_ccf_file = ''
    mutation_ccf_file = ''
    run_tool(indiv, sif_file, mutation_ccf_file, cluster_ccf_file, abundances_file)


if __name__ == '__main__':
    main()

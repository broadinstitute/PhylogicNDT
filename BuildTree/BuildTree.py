import sys

sys.path.append('../')

import logging

logging.basicConfig(filename='build_tree.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=getattr(logging, "DEBUG"))

# Main run method


def run_tool(args):
    logging.debug('Arguments {}'.format(args))

    import data.Patient as Patient
    from CellPopulationEngine import CellPopulationEngine
    from BuildTreeEngine import BuildTreeEngine
    from Clustering_Results import Clustering_Results

    # init a Patient
    patient_data = Patient.Patient(indiv_name=args.indiv_id, driver_genes_file=args.driver_genes_file)

    # Patient load cluster and mut ccf files
    clustering_results = Clustering_Results(args.mutation_ccf_file, args.cluster_ccf_file)
    patient_data.ClusteringResults = clustering_results

    bt_engine = BuildTreeEngine(patient_data)

    bt_engine.build_tree(args.n_iter)

    patient_data.TopTree = bt_engine.top_tree
    patient_data.TreeEnsemble = bt_engine.trees

    cp_engine = CellPopulationEngine(patient_data)
    constrained_ccf = cp_engine.samples_average_constrained_ccf(args.n_iter)
    cell_ancestry = bt_engine.get_cell_ancestry()
    cell_abundance = cp_engine.compute_cell_abundance(constrained_ccf, cell_ancestry)

    # Output and visualization
    import output.PhylogicOutput
    phylogicoutput = output.PhylogicOutput.PhylogicOutput()
    phylogicoutput.write_tree_tsv(bt_engine.mcmc_trace, bt_engine.trees_ll, args.indiv_id)
    phylogicoutput.write_constrained_ccf_tsv(constrained_ccf, cell_ancestry, args.indiv_id)
    phylogicoutput.write_cell_abundances_tsv(cell_abundance, cell_ancestry, args.indiv_id)
    phylogicoutput.generate_html_from_tree(args.mutation_ccf_file, args.cluster_ccf_file, args.indiv_id + '_build_tree_posteriors.tsv',
                                           args.indiv_id + '_constrained_CCF.tsv', sif=args.sif, drivers=patient_data.driver_genes,
                                           treatment_file=args.treatment_data)

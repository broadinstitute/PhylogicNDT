#!/usr/bin/env python
# PhylogicNDT
# Copyright (c) 2015-2019, Broad Institute, Inc. and The General Hospital Corporation. All rights reserved.
# Copyright (c) 2015-2018,  Ignaty Leshchiner, Dimitri Livitz, Daniel Rosebrock, Gad Getz. All rights reserved.
# Copyright (c) 2018-2019, Ignaty Leshchiner, Liudmila Elagina, Justin Cha, Oliver Spiro, Aina Martinez, Gad Getz. All rights reserved.

import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/")

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'phylogicndt.log')
print(filename)
logging.basicConfig(filename=filename,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=getattr(logging, "INFO"))

import Cluster.Cluster  # Cluster Tool
import PhylogicSim.Simulations
import BuildTree.BuildTree  # Tree Building Tool
import BuildTree.CellPopulation
import GrowthKinetics.GrowthKinetics
import SinglePatientTiming.SinglePatientTiming
import LeagueModel.LeagueModel


def build_parser():
    parser = argparse.ArgumentParser(description="Run PhylogicNDT")

    # global parameters common to most tools
    base_parser = argparse.ArgumentParser(add_help=False)

    # option for specifying individual/patient ID
    base_parser.add_argument('--indiv_id', '-i',
                             type=str,
                             action='store',
                             dest='indiv_id',
                             default='Indiv1',
                             help='Patient/Case ID')

    # Samples information
    # Specifying samples on cmdline one-by-one in format sample_id:maf_fn:seg_fn:purity:timepoint
    base_parser.add_argument("-s", "--sample", dest='sample_data', action='append', type=str,
                             help="Sample data, format sample_id:maf_fn:seg_fn:purity:timepoint; each sample specify separately by multiple -s ..; ")

    # Alternative: Instead of specifying samples on cmdline, a tsv - sif (sample information file) may be used.
    base_parser.add_argument("-sif", "--sample_information_file", dest='sif', type=str, help="""Sample information tsv file with sample_ids and CCF and copy-number file_paths; \n 
    format per row (with header) sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint""")

    # Filtering of Mutations/Events
    # option for specifying blacklist
    base_parser.add_argument("-bl", '--artifact_blacklist',
                             type=str,
                             action='store',
                             dest='artifact_blacklist',
                             default=os.path.join(os.path.dirname(__file__), 'data/supplement_data/Blacklist_SNVs.txt'),
                             help='path to blacklist')
    # whitelist - always overrules blacklist.
    base_parser.add_argument("-wl", '--artifact_whitelist',
                             type=str,
                             action='store',
                             dest='artifact_whitelist',
                             default='',
                             help='path to artifact whitelist - takes precedence over blacklist')

    # option for specifying custom drivers list
    base_parser.add_argument("-drv", '--driver_genes_file',
                             type=str,
                             action='store',
                             dest='driver_genes_file',
                             default=os.path.join(os.path.dirname(__file__),
                                                  'data/supplement_data/Driver_genes_v1.0.txt'),
                             help='driver list file')

    base_parser.add_argument("-tr", '--treatment_data',
                             type=str,
                             action='store',
                             dest='treatment_data',
                             default=None,
                             help='path to treatment data file')

    base_parser.add_argument('-ts', '--tumor_size',
                             action='store',
                             dest='tumor_size',
                             default=None)

    base_parser.add_argument('--blacklist_threshold', '-bt',
                             type=float,
                             action='store',
                             dest='blacklist_threshold',
                             default=0.1,
                             help='ccf threshold for blacklisting clusters for a BuildTree and Cell Population')

    base_parser.add_argument('--seed',
                             type=int,
                             action='store',
                             dest='seed',
                             default=None,
                             help='input a random seed for reproducibility')

    # different Tools of the PhylogicNDT Package
    subparsers = parser.add_subparsers(title="tool", description="Choose a tool to run", help='Try the Cluster tool')

    # Cluster - multidimensional clustering of mutations in CCF space
    clustering = subparsers.add_parser("Cluster",
                                       help="Cluster Mutation's CCFs (from one or multiple samples) to define clonal expansions",
                                       parents=[base_parser])

    # run Cluster and BuildTree  together
    clustering.add_argument('--run_with_BuildTree', '-rb',
                            action="store_true",
                            dest='buildtree',
                            help='Run the BuildTree Module right after clustering and generate joint report')

    # option for specifying PoN - will be used to append to blacklist.
    clustering.add_argument('--PoN',
                            type=str,
                            action='store',
                            dest='PoN',
                            default="false",
                            help='PoN to use, specify false to skip.')

    # Overwrite Blacklist
    clustering.add_argument('--Delete_Blacklist',
                            action="store_true",
                            dest='Delete_Blacklist',
                            help='Generate new blacklist from PoN')

    ##Clustering Parameters

    # num ccf bins
    clustering.add_argument('--grid_size', '-g',
                            type=int,
                            action='store',
                            dest='grid_size',
                            default=101,
                            help='num ccf bins, must match for txt input, otherwise may be any number that grid in absolute RData is divisble by.')

    # num iterations
    clustering.add_argument('--n_iter', '-ni',
                            type=int,
                            action='store',
                            dest='iter',
                            default=250,
                            help='number iterations')

    # Cluster with Indels?
    clustering.add_argument('--use_indels',
                            action="store_true",
                            dest='use_indels',
                            help='Use indels in clustering. By default indels are added in after clustering.')
    # Impute missing?
    clustering.add_argument('--impute',
                            action="store_true",
                            dest='impute_missing',
                            help='Assume 0 ccf for missing mutations.')

    # Don't use poorly clustered mutations.
    clustering.add_argument('--min_coverage', '-mc',
                            type=int,
                            action='store',
                            dest='min_cov',
                            default=8,
                            help='Mutations with coverage lower than this will not be used to cluster and instead re-assigned after dp clustering')

    clustering.add_argument('--cancer_type', '-ct',
                            type=str,
                            action='store',
                            dest='cancer_type',
                            default='All_cancer',
                            help='cancer type -- useful for calling focal events')

    clustering.add_argument('--cn_peaks',
                            type=str,
                            action='store',
                            dest='gistic_fn',
                            default=None,
                            help='interval file with focal amp and del regions specified')

    clustering.add_argument('--Pi_k_r',
                            type=int,
                            action='store',
                            dest='Pi_k_r',
                            default=3,
                            help='parameter r of the negative binomial prior over number of clusters')

    clustering.add_argument('--Pi_k_mu',
                            type=int,
                            action='store',
                            dest='Pi_k_mu',
                            default=3,
                            help='parameter mu of the negative binomial prior over number of clusters')

    clustering.add_argument('--order_by_timepoint',
                            action='store_true',
                            dest='order_by_timepoint',
                            help='Order samples by timepoint values as specified in .sif or cmdline')

    # output type
    clustering.add_argument('--maf',
                            action="store_true",
                            dest='maf',
                            help='output maf if set')

    clustering.add_argument('--no_html',
                            action="store_false",
                            dest='html',
                            help='output html if set')
    clustering.add_argument('--time_points',
                            action='store',
                            dest='time_points',
                            default=None)
    clustering.add_argument('--scale',
                            action='store_true',
                            dest='scale',
                            default=False)

    clustering.set_defaults(func=Cluster.Cluster.run_tool)

    # num samples to use - can be set to restrict to first n samples.
    clustering.add_argument('--use_first_n_samples', '-ns',
                            type=int,
                            action='store',
                            dest='n_samples',
                            default=0,
                            help='num samples to match, 0 for all samples')

    clustering.add_argument('--maf_input_type', '-mt',
                            action='store',
                            dest='maf_input_type',
                            default='auto')

    # BuildTree  Tool

    buildtree = subparsers.add_parser("BuildTree", help="BuildTree module for constructing of phylogenetic trees.",
                                      parents=[base_parser])
    buildtree.add_argument('--cluster_ccf', '-c',
                           type=str,
                           action='store',
                           dest='cluster_ccf_file',
                           help='tsv file phylogic clustering results')
    buildtree.add_argument('--mutation_ccf', '-m',
                           type=str,
                           action='store',
                           dest='mutation_ccf_file',
                           help='tsv file generated by clustering')
    buildtree.add_argument('--n_iter', '-ni',
                           type=int,
                           action='store',
                           dest='n_iter',
                           default=250,
                           help='number iterations')
    # Specifying cluster ids to blacklist from BuildTree and CellPopulation
    buildtree.add_argument("-bc", "--blacklist_cluster",
                            dest='blacklist_cluster',
                            action='append',
                            type=str,
                            help="List cluster ids to blacklist from BuildTree and CellPopulation")

    buildtree.set_defaults(func=BuildTree.BuildTree.run_tool)

    # CellPopulation  Tool

    cellpopulation = subparsers.add_parser("CellPopulation", help="CellPopulation module for computing cell abundance.",
                                           parents=[base_parser])
    cellpopulation.add_argument('--cluster_ccf', '-c',
                                type=str,
                                action='store',
                                dest='cluster_ccf_file',
                                help='tsv file phylogic clustering results')
    cellpopulation.add_argument('--mutation_ccf', '-m',
                                type=str,
                                action='store',
                                dest='mutation_ccf_file',
                                help='tsv file generated by clustering')
    cellpopulation.add_argument('--tree_tsv', '-t',
                                type=str,
                                action='store',
                                dest='tree_tsv',
                                help='tsv file generated by build tree module')
    cellpopulation.add_argument('--n_iter', '-ni',
                                type=int,
                                action='store',
                                dest='n_iter',
                                default=250,
                                help='number iterations')
    cellpopulation.set_defaults(func=BuildTree.CellPopulation.run_tool)

    # GrowthKinetics  Tool

    growthkinetics = subparsers.add_parser("GrowthKinetics",
                                           help="Sample growth rates and fitness across ensemble of trees", parents=[base_parser])    
    growthkinetics.add_argument('--abundance_mcmc_trace', '-ab',
                                type=str,
                                action='store',
                                dest='abundance_mcmc_trace',
                                help='tsv file generated by CellPopulation')    
    growthkinetics.add_argument('--n_iter', '-ni',
                                type=int,
                                action='store',
                                dest='n_iter',
                                default=250,
                                help='number iterations')
    growthkinetics.add_argument('--wbc', '-w',
                                type=int,
                                nargs='+',
                                default=[],
                                action='store',
                                dest='wbc',
                                help='wbc')
    growthkinetics.add_argument('--time', '-t',
                                type=int,
                                nargs='+',
                                default=[],
                                action='store',
                                dest='time',
                                help='time')

    growthkinetics.set_defaults(func=GrowthKinetics.GrowthKinetics.run_tool)

    # PhylogicSim simulator
    simulations = subparsers.add_parser("PhylogicSim", help="Generate simulations drawn from a truth clustering.",
                                        parents=[base_parser])
    simulations.add_argument('-p',
                             type=float,
                             action='store',
                             dest='purity',
                             default=0.7,
                             help='Purity of tumor sample, from 0.0 to 1.0')
    simulations.add_argument('-a',
                             type=float,
                             action='store',
                             dest='artifacts',
                             default=0.0,
                             help='Fraction of mutations that are artifacts(random af). Value from 0.0 to 1.0')
    simulations.add_argument('-nm',
                             type=int,
                             action='store',
                             dest='nmuts',
                             default=500,
                             help='Number of mutations. WES recommended 300, WGS recommended 1500')
    simulations.add_argument('-ns',
                             type=int,
                             action='store',
                             dest='nsamp',
                             default=5,
                             help='Number of samples')
    simulations.add_argument('-nodes',
                             type=int,
                             action='store',
                             dest='min_nodes',
                             default=4,
                             help='Number of distinct nodes in tree - clusters')
    simulations.add_argument('-cov',
                             type=str,
                             action='store',
                             dest='cov_file',
                             default=None,
                             help='File of total coverage to sample from. Each line represents a single coverage value to sample. See example.')
    simulations.add_argument('-seg',
                             type=str,
                             action='store',
                             dest='cn_dist',
                             default=None,
                             help='Segment file with copy number distribution. See example for format.')
    simulations.add_argument('-clust_file',
                             type=str,
                             action='store',
                             dest='clust_file',
                             default=None,
                             help='File of clusters and ccfs if we want to force their values. See example for format.')
    simulations.add_argument('-ap',
                             type=float,
                             action='store',
                             dest='alpha',
                             default=2,
                             help='Alpha parameter for the betabinomial to determine coverage')
    simulations.add_argument('-b',
                             type=float,
                             action='store',
                             dest='beta',
                             default=18,
                             help='Beta parameter for the betabinomial to determine coverage')
    simulations.add_argument('-nb',
                             type=int,
                             action='store',
                             dest='nbin',
                             default=1000,
                             help='N parameter for the betabinomial to determine coverage')
    simulations.add_argument('-pfile',
                             type=str,
                             action='store',
                             dest='purity_file',
                             default=None,
                             help='TSV File of purity values for each sample, and optionally a 2/3/4 column with a, b and n values. Number of samples needs to match ns.')
    simulations.set_defaults(func=PhylogicSim.Simulations.run_tool)

    timing = subparsers.add_parser("Timing", help="Time somatic events in one or multiple samples.",
                                   parents=[base_parser])
    timing.add_argument('-min_supporting_muts',
                        type=int,
                        action='store',
                        dest='min_supporting_muts',
                        default=3,
                        help='Minimum number of supporting mutations to time a copy number event')
    timing.set_defaults(func=SinglePatientTiming.SinglePatientTiming.run_tool)

    single_patient_timing = subparsers.add_parser("SinglePatientTiming", help="Time somatic events in one or multiple samples.",
                                   parents=[base_parser])
    single_patient_timing.add_argument('-min_supporting_muts',
                        type=int,
                        action='store',
                        dest='min_supporting_muts',
                        default=3,
                        help='Minimum number of supporting mutations to time a copy number event')
    single_patient_timing.set_defaults(func=SinglePatientTiming.SinglePatientTiming.run_tool)

    leaguemodel = subparsers.add_parser("LeagueModel", help="Time somatic events across a cohort.",
                                         parents=[base_parser])
    leaguemodel.add_argument('--cohort', '-cohort',
                             type=str,
                             action='store',
                             dest='cohort',
                             default='None',
                             help='cohort name')
    leaguemodel.add_argument('--comps', '-comps',
                             type=str,
                             nargs='+',
                             action='store',
                             dest='comps',
                             help='all comparison inputs')
    leaguemodel.add_argument('--comparison_fn', '-comparison_fn',
                             type=str,
                             action='store',
                             dest='comparison_fn',
                             default=None,
                             help='comparison file')
    leaguemodel.add_argument('--n_perms', '-n_perms',
                             type=int,
                             action='store',
                             dest='n_perms',
                             default='500',
                             help='number of permutations')
    leaguemodel.add_argument('--n_seasons', '-n_seasons',
                             type=int,
                             action='store',
                             dest='n_seasons',
                             default=200,
                             help='number of seasons')
    leaguemodel.add_argument('--percent_subset', '-percent_subset',
                             type=float,
                             action='store',
                             dest='percent_subset',
                             default=0.8,
                             help='percent samples to subset in each permutation of league model')
    leaguemodel.add_argument('--keep_samps_w_event', '-keep_samps_w_event',
                             type=str,
                             action='store',
                             dest='keep_samps_w_event',
                             default=None,
                             help='keep only samples with specified event')
    leaguemodel.add_argument('--remove_samps_w_event', '-remove_samps_w_event',
                             type=str,
                             action='store',
                             dest='remove_samps_w_event',
                             default=None,
                             help='keep only samples with specified event')
    leaguemodel.add_argument('--force_final_event_list', '-force_final_event_list',
                             type=str,
                             action='store',
                             dest='force_final_event_list',
                             default=None,
                             help='force_final_event_list')
    leaguemodel.add_argument('--split_fn', '-split_fn',
                             type=str,
                             action='store',
                             dest='split_fn',
                             default=None,
                             help='split file')
    leaguemodel.add_argument('--split_flag', '-split_flag',
                             type=str,
                             action='store',
                             dest='split_flag',
                             default=None,
                             help='split file')
    leaguemodel.add_argument('--final_sample_list', '-final_sample_list',
                             type=str,
                             action='store',
                             dest='final_sample_list',
                             default=None,
                             help='final sample list file')
    leaguemodel.add_argument('--num_games_against_each_opponent', '-num_games_against_each_opponent',
                             type=int,
                             action='store',
                             dest='num_games_against_each_opponent',
                             default=2,
                             help='number of games each opponent plays against another in a season')
    leaguemodel.set_defaults(func=LeagueModel.LeagueModel.run_league_model)

    # print help without -h
    if len(sys.argv) < 2: parser.print_help(sys.stderr)
    return parser.parse_args()


# if __name__ == "main":
parser = build_parser()
args = parser
args.func(args)

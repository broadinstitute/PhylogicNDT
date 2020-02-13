import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import itertools
from io import BytesIO
import base64
from string import Template
import functools
import scipy.stats
import random
import math
import re
import networkx as nx

import matplotlib

# turn off the display var for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.dom.minidom as dom

import BuildTree.Tree


class PhylogicOutput(object):
    # Static class for Phylogic visualization and serialization

    def __init__(self):
        pass

    def generate_html_from_clustering_results(self, results, patient, drivers=(), treatment_file=None):
        patient_id = patient.indiv_name
        sample_names = [s.sample_name for s in patient.sample_list]
        time_points = []
        for sample in patient.sample_list:
            if isinstance(sample.timepoint, int) or isinstance(sample.timepoint, float):
                time_points.append(sample.timepoint)
            else:
                time_points = []
                break
        if len(set(time_points)) < len(time_points):
            time_points = []
        cluster_dict = {c: {'muts': {}, 'cnvs': {}} for c in range(1, len(results.clust_CCF_dens) + 1)}
        n_muts = dict.fromkeys(range(1, len(results.clust_CCF_dens) + 1), 0)
        c_drivers = {c: [] for c in range(1, len(results.clust_CCF_dens) + 1)}
        for i, sample in enumerate(patient.sample_list):
            for mut in sample.concordant_variants:
                if hasattr(mut, 'cluster_assignment') and mut.cluster_assignment:
                    c = mut.cluster_assignment
                    if mut.type == 'CNV':
                        mut_name = mut.event_name
                    elif mut.gene and mut.prot_change:
                        mut_name = '_'.join((mut.gene, mut.prot_change))
                    elif mut.gene:
                        mut_name = ':'.join((mut.gene, str(mut.pos), mut.ref, mut.alt))
                    else:
                        mut_name = ':'.join((str(mut.chrN), str(mut.pos), mut.ref, mut.alt))
                    ccf_hat, ccf_high, ccf_low = self._get_mean_high_low(np.array(mut.ccf_1d).astype(float))
                    alt_cnt = mut.alt_cnt
                    ref_cnt = mut.ref_cnt
                    if mut.type != 'CNV':
                        cluster_dict[c]['muts'].setdefault(mut_name, {'ccf_hat': [], 'alt_cnt': [], 'ref_cnt': []})
                        cluster_dict[c]['muts'][mut_name]['ccf_hat'].append(ccf_hat)
                        cluster_dict[c]['muts'][mut_name]['alt_cnt'].append(alt_cnt)
                        cluster_dict[c]['muts'][mut_name]['ref_cnt'].append(ref_cnt)
                        cluster_dict[c]['muts'][mut_name]['chrom'] = mut.chrN
                        cluster_dict[c]['muts'][mut_name]['pos'] = mut.pos
                        if len(sample_names) == 1:
                            cluster_dict[c]['muts'][mut_name]['ccf_dist'] = list(map(float, mut.ccf_1d))
                        if i == 0:
                            n_muts[c] += 1
                            if mut_name.split('_')[0] in drivers:
                                c_drivers[c].append(mut_name)
                    else:
                        cluster_dict[c]['cnvs'].setdefault(mut_name, {'ccf_hat': [], 'alt_cnt': [], 'ref_cnt': []})
                        cluster_dict[c]['cnvs'][mut_name]['ccf_hat'].append(ccf_hat)
                        cluster_dict[c]['cnvs'][mut_name]['alt_cnt'].append(alt_cnt)
                        cluster_dict[c]['cnvs'][mut_name]['ref_cnt'].append(ref_cnt)
                        cluster_dict[c]['cnvs'][mut_name]['chrom'] = mut.chrN
                        cluster_dict[c]['cnvs'][mut_name]['pos'] = ''
        for i, ccf_hists in enumerate(results.clust_CCF_dens):
            c = i + 1
            cluster_dict[c]['color'] = ClusterColors.get_rgb_string(c)
            cluster_dict[c]['line_width'] = min(12, math.ceil(3 * n_muts[c] / 7.))
            cluster_dict[c]['drivers'] = c_drivers[c]
            cluster_dict[c].setdefault('ccf_hat', [])
            cluster_dict[c].setdefault('ccf_high', [])
            cluster_dict[c].setdefault('ccf_low', [])
            if len(sample_names) == 1:
                cluster_dict[c]['ccf_dist'] = list(map(float, ccf_hists[0]))
            for ii, sample in enumerate(patient.sample_list):
                ccf_hat, ccf_high, ccf_low = self._get_mean_high_low(ccf_hists[ii])
                cluster_dict[c]['ccf_hat'].append(ccf_hat)
                cluster_dict[c]['ccf_high'].append(ccf_high)
                cluster_dict[c]['ccf_low'].append(ccf_low)

        if treatment_file and time_points:
            treatment_data = []
            with open(treatment_file, 'r') as f:
                header = f.readline().strip('\n\r').split('\t')
                for line in f:
                    fields = dict(zip(header, line.strip('\n\r').split('\t')))
                    treatment_data.append({'tx': fields['tx'],
                                           'tx_start': float(fields['tx_start']),
                                           'tx_end': float(fields['tx_end']) if fields['tx_end'] else time_points[-1]})
        else:
            treatment_data = None

        self.write_html_report(patient_id, sample_names, cluster_dict, time_points=time_points,
                               treatment_data=treatment_data)

    def generate_html_from_tree(self, mutation_ccf_file, cluster_ccf_file, tree, abundances, sif=None, drivers=(),
                                treatment_file=None, tumor_sizes_file=None, cnv_file=None):
        """
        Creates html report from Clustering and BuildTree output files
        Args:
            mutation_ccf_file: path to mut_ccfs file
            cluster_ccf_file: path to cluster_ccfs file
            tree: path to tree_tsv file
            abundances: path to constrained_ccfs file
            sif: path to sample information file
            drivers: list of drivers
            treatment_file: path to treatment file
            tumor_sizes_file: path to tumor sizes file
            cnv_file: path to cnvs file

        """
        sample_names = []
        treatment_data = None
        if sif:
            timepoints = []
            with open(sif, 'r') as sif_file:
                header = sif_file.readline().strip('\n\r').lower().split('\t')
                for line in sif_file:
                    fields = dict(zip(header, line.strip('\n\r').split('\t')))
                    sample_names.append(fields['sample_id'])
                    timepoints.append(float(fields['timepoint']) if fields['timepoint'] else np.nan)
            if np.all(~np.isnan(timepoints)) and len(set(timepoints)) == len(sample_names):
                sorting_idx = sorted(range(len(sample_names)), key=lambda k: timepoints[k])
                sample_names = [sample_names[i] for i in sorting_idx]
                timepoints = [timepoints[i] for i in sorting_idx]
                if treatment_file:
                    treatment_data = []
                    with open(treatment_file, 'r') as f:
                        header = f.readline().strip('\n\r').split('\t')
                        for line in f:
                            fields = dict(zip(header, line.strip('\n\r').split('\t')))
                            treatment_data.append({'tx': fields['tx'],
                                                   'tx_start': float(fields['tx_start']),
                                                   'tx_end': float(fields['tx_end']) if fields['tx_end'] else timepoints[-1]})
                if tumor_sizes_file:
                    with open(tumor_sizes_file, 'r') as f:
                        # f.readline()
                        tumor_sizes = [list(map(float, line.strip('\n\r').split('\t'))) for line in f]
                else:
                    tumor_sizes = [[t, 1.] for t in timepoints]
            else:
                timepoints = []
                tumor_sizes = [[t, 1.] for t in timepoints]
        else:
            timepoints = []
            tumor_sizes = [[t, 1.] for t in timepoints]
        aliases = ['T' + str(i) for i, s in enumerate(sample_names)]

        cluster_dict = {}
        cluster_ccfs = {}
        patient = ''
        ccf_dist_keys = ['postDP_ccf_' + str(i / 100.) for i in range(101)]
        with open(cluster_ccf_file, 'r') as clust_file:
            header = clust_file.readline().strip('\n\r').split('\t')
            for line in clust_file:
                fields = dict(zip(header, line.strip('\n\r').split('\t')))
                if not patient:
                    patient = fields['Patient_ID']
                c = int(fields['Cluster_ID'])
                sample = fields['Sample_ID']
                ccf_hat = float(fields['postDP_ccf_mean'])
                ccf_high = float(fields['postDP_ccf_CI_high'])
                ccf_low = float(fields['postDP_ccf_CI_low'])
                cluster_dict.setdefault(c, {'ccf_hat': [], 'ccf_high': [], 'ccf_low': [], 'muts': {}, 'cnvs': {},
                                            'drivers': [], 'tumor_abundance': []})
                cluster_dict[c]['ccf_hat'].append(ccf_hat)
                cluster_dict[c]['ccf_high'].append(ccf_high)
                cluster_dict[c]['ccf_low'].append(ccf_low)
                if sample not in sample_names:
                    sample_names.append(sample)
                ccf_dist = [float(fields[k]) for k in ccf_dist_keys]
                cluster_ccfs.setdefault(sample, {})
                cluster_ccfs[sample][c] = np.array(ccf_dist)
                if len(sample_names) == 1:
                    cluster_dict[c]['ccf_dist'] = ccf_dist
        n_muts = dict.fromkeys(cluster_dict, 0)
        ccf_dist_keys = ['preDP_ccf_'+str(i / 100.) for i in range(101)]
        with open(mutation_ccf_file, 'r') as mut_file:
            header = mut_file.readline().strip('\n\r').split('\t')
            for line in mut_file:
                fields = dict(zip(header, line.strip('\n\r').split('\t')))
                sample = fields['Sample_ID']
                hugo = fields['Hugo_Symbol']
                prot_change = fields['Protein_change']
                chrom = fields['Chromosome']
                pos = fields['Start_position']
                ref = fields['Reference_Allele']
                alt = fields['Tumor_Seq_Allele']
                ref_cnt = fields['t_ref_count']
                alt_cnt = fields['t_alt_count']
                c = int(fields['Cluster_Assignment'])
                ccf_hat = float(fields['preDP_ccf_mean'])
                if fields['Variant_Type'] == 'CNV':
                    mut_name = hugo
                elif hugo != 'Unknown' and prot_change:
                    mut_name = '_'.join((hugo, prot_change))
                elif hugo != 'Unknown':
                    mut_name = ':'.join((hugo, pos, ref, alt))
                else:
                    mut_name = ':'.join((chrom, pos, ref, alt))
                cluster_dict[c]['muts'].setdefault(mut_name, {'ccf_hat': [], 'alt_cnt': [], 'ref_cnt': []})
                cluster_dict[c]['muts'][mut_name]['ccf_hat'].append(ccf_hat)
                cluster_dict[c]['muts'][mut_name]['alt_cnt'].append(alt_cnt)
                cluster_dict[c]['muts'][mut_name]['ref_cnt'].append(ref_cnt)
                cluster_dict[c]['muts'][mut_name]['chrom'] = chrom
                cluster_dict[c]['muts'][mut_name]['pos'] = pos
                if len(sample_names) == 1:
                    cluster_dict[c]['muts'][mut_name]['ccf_dist'] = [float(fields[k]) for k in ccf_dist_keys]
                if sample == sample_names[0]:
                    n_muts[c] += 1
                    if hugo in drivers:
                        cluster_dict[c]['drivers'].append(mut_name)
        if cnv_file:
            unique_cnvs = {s: set() for s in sample_names}
            with open(cnv_file, 'r') as cnvs:
                header = cnvs.readline().strip('\n\r').split('\t')
                for line in cnvs:
                    fields = dict(zip(header, line.strip('\n\r').split('\t')))
                    sample = fields['Sample_ID']
                    mut_name = fields['Event_Name']
                    if mut_name in unique_cnvs[sample]:
                        mut_name += '_'
                    unique_cnvs[sample].add(mut_name)
                    ccf_hat = float(fields['preDP_ccf_mean'])
                    c = int(fields['Cluster_Assignment'])
                    chrom = fields['Chromosome']
                    cluster_dict[c]['cnvs'].setdefault(mut_name, {'ccf_hat': []})
                    cluster_dict[c]['cnvs'][mut_name]['ccf_hat'].append(ccf_hat)
                    cluster_dict[c]['cnvs'][mut_name]['chrom'] = chrom
                    cluster_dict[c]['cnvs'][mut_name]['pos'] = ''
        for c in cluster_dict:
            cluster_dict[c]['line_width'] = min(12, math.ceil(3 * n_muts[c] / 7.))
            cluster_dict[c]['color'] = ClusterColors.get_rgb_string(c)
        edges_list = []
        tree_iterations = []
        # Removed eval when reading file
        with open(tree, 'r') as tree_fh:
            header = tree_fh.readline().strip('\n\r').split('\t')
            for line in tree_fh:
                row = line.strip('\n\r').split('\t')
                edges = self.reformat_edges_for_input(row[header.index('edges')])
                edges_list.append(edges)
                tree_iterations.append(row[header.index('n_iter')])
        child_dicts = []
        for i, e in enumerate(edges_list):
            child_dicts.append({n: [] for n in itertools.chain(*e)})
            # child_dicts[-1].update({None: [1]})
            for parent, child in e:
                child_dicts[-1][parent].append(child)
        edges = edges_list[0]
        tree = BuildTree.Tree.Tree()
        for identifier in set(itertools.chain(*edges)):
            if identifier:
                tree.add_node(identifier)
        tree.add_edges(edges)
        tree.set_root(tree.get_node_by_id(1))
        constrained_ccfs = {c: {} for c in tree.nodes.keys() if c}
        cell_abundances = {s: {} for s in sample_names}
        with open(abundances, 'r') as abun_fh:
            header = abun_fh.readline().strip('\n\r').split('\t')
            for line in abun_fh:
                fields = dict(zip(header, line.strip('\n\r').split('\t')))
                c = int(re.search(r'(\d{1,2})$', fields['Cell_population']).group(1))
                cell_abundance = float(fields['Constrained_CCF'])
                sample_name = fields['Sample_ID']
                cell_abundances[sample_name][c] = cell_abundance
                constrained_ccfs[c][sample_name] = (cell_abundance / 100.)
        for c in constrained_ccfs:
            cluster_dict[c]['tumor_abundance'] = [constrained_ccfs[c][s] for s in sample_names]
        pie_plot_dir = '{}_pie_plots'.format(patient)
        if not os.path.exists(pie_plot_dir):
            os.mkdir(pie_plot_dir)
        pie_plots = [self.make_pie_plot(tree, cell_abundances[s], outdir=pie_plot_dir, sample=s) for s in sample_names]

        self.write_html_report(patient, sample_names, cluster_dict, aliases=aliases, time_points=timepoints,
                               pie_plots=pie_plots, child_dicts=child_dicts, tumor_sizes=tumor_sizes,
                               treatment_data=treatment_data, tree_iterations=tree_iterations)

    @staticmethod
    def _get_abundances(cluster_ccfs, tree, n_iter=100):
        '''

        Args:
            cluster_ccfs:
            tree:
            n_iter:

        Returns:

        '''
        abundances = dict.fromkeys(tree.nodes, 0)
        reducing_function = lambda a, b: a - iter_ccf[b] if b in iter_ccf else a
        for i in range(n_iter):
            iter_ccf = {1: 100}
            for level, level_nodes in enumerate(tree.traverse_by_level()):
                parent_child_tuples = list(
                    itertools.chain(*(itertools.product([node.identifier], node.children) for node in level_nodes)))
                random.shuffle(parent_child_tuples)
                for parent, child in parent_child_tuples:
                    cluster_ccf = cluster_ccfs[child]
                    siblings = tree.get_node_by_id(child).siblings
                    reduced_parent_abundance = functools.reduce(reducing_function, siblings, iter_ccf[parent])
                    reduced_ccf = np.concatenate(
                        [cluster_ccf[:reduced_parent_abundance + 1], np.zeros(100 - reduced_parent_abundance)])
                    ccf_sum = sum(reduced_ccf)
                    if ccf_sum > 0:
                        normalized_ccf = reduced_ccf / ccf_sum
                        custm = scipy.stats.rv_discrete(name='custm', values=(np.arange(101.), normalized_ccf))
                        sampled_abundance = custm.rvs(size=1)[0]
                        iter_ccf[child] = sampled_abundance
                        abundances[child] += sampled_abundance
                    else:
                        iter_ccf[child] = 0
        for node in abundances:
            abundances[node] /= float(n_iter)
        abundances[1] = 100
        return abundances

    @classmethod
    def write_html_report(cls, patient, sample_names, cluster_dict, aliases=None, time_points=None, pie_plots=None,
                          child_dicts=None, treatment_data=None, tumor_sizes=None, tree_iterations=None):
        """
        Writes the html report
        Args:
            patient: Patient ID
            sample_names: List of sample IDs
            cluster_dict: {cluster#: {
                            "ccf_hat": list,
                            "ccf_high": list,
                            "ccf_low": list,
                            "line_width": num,
                            "color": "rgb({r},{g},{b})",
                            "drivers": list,
                            "muts": {mut_str: {"ccf_hat": list, "alt_cnt": list, "ref_cnt": list}}
                            }
                          }
            aliases: List of sample aliases
            time_points: List of sample time points
            pie_plots: List of base64-encoded pngs for pie plots
            child_dicts: List of dicts mapping parent to list of children (with None mapping to [trunk])
            treatment_data: List of treatments of the format {"tx": str, "tx_start": num, "tx_end": num}
            tumor_sizes: List of date/tumor size pairs
            tree_iterations: List of number of iterations for each tree

        Returns:
            None

        """
        sample_name_table_options = {'sample_names': sample_names}
        time_switch_options = {'show': False}
        cluster_switches_options = {'xlabels': sample_names, 'cluster_dict': cluster_dict}
        clustering_plot_options = {'xlabels': sample_names, 'cluster_dict': cluster_dict}
        tree_options = False
        tree_switch_options = False
        treatment_plot_options = False
        mutation_visual_options = {'aliases': sample_names, 'cluster_dict': cluster_dict}
        cluster_table_options = {'aliases': sample_names, 'cluster_dict': cluster_dict}
        mutation_plot_options = {'xlabels': sample_names, 'cluster_dict': cluster_dict}
        fish_plot_options = False

        if time_points and child_dicts and tumor_sizes:
            fish_plot_options = {'xlabels': sample_names, 'time_points': time_points,
                                 'tumor_sizes': tumor_sizes, 'cluster_dict': cluster_dict,
                                 'child_dict': child_dicts[0]}
            cluster_switches_options['child_dict'] = child_dicts[0]
            cluster_switches_options['tumor_sizes'] = tumor_sizes

        columns = ['Sample Name']
        if aliases:
            columns.append('Alias')
            sample_name_table_options['sample_aliases'] = aliases
            clustering_plot_options['xlabels'] = aliases
            cluster_switches_options['xlabels'] = aliases
            mutation_visual_options['aliases'] = aliases
            cluster_table_options['aliases'] = aliases
            mutation_plot_options['xlabels'] = aliases
            if fish_plot_options:
                fish_plot_options['xlabels'] = aliases

        if time_points:
            columns.append('Date')
            sample_name_table_options['sample_dates'] = time_points
            time_switch_options['show'] = time_points != list(range(len(sample_names))) and len(np.unique(time_points)) > 1
            cluster_switches_options['time_points'] = time_points
            clustering_plot_options['time_points'] = time_points

        columns.append('Clonal Abundance')
        if pie_plots:
            sample_name_table_options['pie_plots'] = pie_plots

        if child_dicts and tree_iterations:
            tree_coordinates_list = []
            print('Plotting trees')
            n_trees = 0
            for child_dict, ni in zip(child_dicts, tree_iterations):
                if ni < 10:
                    break
                n_trees += 1
                n_muts = dict.fromkeys(child_dict, 0)
                dist_from_parent = dict.fromkeys(itertools.chain(*child_dict.values()))
                for parent in child_dict:
                    if parent is not None:
                        n_muts[parent] += len(cluster_dict[parent]['muts'])
                    for child in child_dict[parent]:
                        if parent is None:
                            dist_from_parent[child] = 200.
                        else:
                            n_muts[child] += len(cluster_dict[parent]['muts'])
                            n_muts_parent = n_muts[parent]
                            diff_muts_child = len(cluster_dict[child]['muts'])
                            jaccard_dist = float(diff_muts_child) / (n_muts_parent + diff_muts_child)
                            dist_from_parent[child] = 170. * math.log(99. * jaccard_dist + 1., 100.) + 30.
                tree_coordinates = cls.get_tree_coordinates(child_dict, dist_from_parent)
                tree_coordinates_list.append(tree_coordinates)

            tree_options = {'patient': patient, 'cluster_dict': cluster_dict,
                            'child_dict': child_dicts[0], 'tree_coordinates': tree_coordinates_list[0]}
            if n_trees > 1:
                tree_switch_options = {'patient': patient, 'cluster_dict': cluster_dict, 'child_dicts': child_dicts,
                                       'tree_coordinates': tree_coordinates_list, 'tree_iterations': tree_iterations}

        if treatment_data:
            treatment_plot_options = {'treatments': treatment_data, 'samples': time_points,
                'treatment_color_map': {s: ClusterColors.get_rgb_string(i + len(cluster_dict)) for
                    i, s in enumerate(np.unique([x['tx'] for x in treatment_data]))}}

        sample_name_table_options['columns'] = columns
        with open(os.path.dirname(__file__) + '/report_template.html_', 'r') as fh_in, open(
                patient + '.phylogic_report.html', 'w') as fh_out:
            html_template = fh_in.read()
            fh_out.write(HTMLTemplate(html_template).substitute(**{
                'indiv_id': patient,
                'sample_name_table_options': json.dumps(sample_name_table_options),
                'time_switch_options': json.dumps(time_switch_options),
                'cluster_switches_options': json.dumps(cluster_switches_options),
                'clustering_plot_options': json.dumps(clustering_plot_options),
                'tree_options': json.dumps(tree_options),
                'tree_switch_options': json.dumps(tree_switch_options),
                'treatment_plot_options': json.dumps(treatment_plot_options),
                'mutation_visual_options': json.dumps(mutation_visual_options),
                'cluster_table_options': json.dumps(cluster_table_options),
                'mutation_plot_options': json.dumps(mutation_plot_options),
                'fish_plot_options': json.dumps(fish_plot_options)}))

    @staticmethod
    def make_pie_plot(tree, cluster_abundances, outdir='', sample=''):
        """
        Makes clonal abundance pie plot
        Args:
            tree: Tree class
            cluster_abundances: dict mapping nodes to sampled abundances
            outdir: output directory for svgs
            sample: sample name

        Returns:
            Base-64 encoded png of the pie plot
        """
        plt.figure(figsize=(1, 1))
        ax = plt.gca()
        pie_slices = {}
        node_order = list(tree.traverse_by_branch())
        for level, clusters in enumerate(tree.traverse_by_level()):
            for node in clusters:
                pie_slices[node] = cluster_abundances[node.identifier]
                if node.parent:
                    pie_slices[node.parent] -= cluster_abundances[node.identifier]
            x = []
            colors = []
            for node in node_order:
                if node in pie_slices:
                    x.append(pie_slices[node])
                    colors.append(ClusterColors.get_hex_string(node.identifier))
            ax.pie(x, colors=colors, radius=.9-(.1*level))
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        bytes_io = BytesIO()
        plt.savefig(bytes_io, bbox_inches='tight', pad_inches=0., transparent=True)
        if outdir and sample:
            plt.savefig(os.path.join(outdir, '{}.pie_plot.svg'.format(sample)), format='svg')
        plt.close()
        return base64.b64encode(bytes_io.getvalue()).decode('UTF-8')

    @staticmethod
    def get_tree_coordinates(child_dict, dist_from_parent, n_iter=100):
        ang_range = np.linspace(0, 2 * math.pi, 73)[:-1]

        def get_coords(ang):
            coord_dict = {None: (0., 0.), 1: (0., dist_from_parent[1])}
            clusters = [None]
            while clusters:
                clust = clusters.pop(0)
                clusters.extend(child_dict[clust])
                if clust is None:
                    continue
                for child in child_dict[clust]:
                    x_coord = coord_dict[clust][0] + (dist_from_parent[child] * math.cos(ang[child]))
                    y_coord = coord_dict[clust][1] + (dist_from_parent[child] * math.sin(ang[child]))
                    coord_dict[child] = (x_coord, y_coord)
            return coord_dict

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def cost_function(coord_dict, curr_min=np.inf):
            cost = 0.
            # Cost for distance between nodes
            for c1, c2 in itertools.combinations(coord_dict, 2):
                x1, y1 = coord_dict[c1]
                x2, y2 = coord_dict[c2]
                dist2 = ((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2))
                if c1 is None or c2 is None:
                    cost += 10. / (dist2 + (10. ** -20))
                else:
                    cost += 1. / (dist2 + (10. ** -20))
                if cost > curr_min:
                    return curr_min + 1
            # Cost for crossing edges and angles < 30 degrees
            parent_child_tuples = list(map(lambda k: itertools.product([k[0]], k[1]), child_dict.items()))
            for e1, e2 in itertools.combinations(itertools.chain(*parent_child_tuples), 2):
                if len(set(e1)) != 2 or len(set(e2)) != 2:
                    # Strange Tree
                    continue
                points = set(e1).union(e2)
                if len(points) == 4:
                    A = coord_dict[e1[0]]
                    B = coord_dict[e1[1]]
                    C = coord_dict[e2[0]]
                    D = coord_dict[e2[1]]
                    if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                        cost += 10. ** 20
                else:
                    pivot = next(p for p in e1 if p in e2)
                    points.remove(pivot)
                    points = list(points)
                    if points[0] in child_dict[pivot]:
                        if abs(math.pi - (
                                ang_from_parent[pivot] - ang_from_parent[points[0]])) % math.pi <= math.pi / 6:
                            cost += 10. ** 10
                    elif points[1] in child_dict[pivot]:
                        if abs(math.pi - (
                                ang_from_parent[pivot] - ang_from_parent[points[1]])) % math.pi <= math.pi / 6:
                            cost += 10. ** 10
                    elif abs(ang_from_parent[points[0]] - ang_from_parent[points[1]]) % math.pi <= math.pi / 6:
                        cost += 10. ** 10
                if cost > curr_min:
                    return curr_min
            return cost

        ang_from_parent = dict.fromkeys(dist_from_parent, 0.)
        ang_from_parent[1] = math.pi / 2
        for i in range(n_iter):
            for clust in ang_from_parent:
                if clust == 1:
                    continue
                ang_update = dict(ang_from_parent)
                min_cost = np.inf
                min_cost_ang = 0
                for a in ang_range:
                    ang_update[clust] = a
                    curr_cost = cost_function(get_coords(ang_update), curr_min=min_cost)
                    if curr_cost < min_cost:
                        min_cost = curr_cost
                        min_cost_ang = a
                ang_from_parent[clust] = min_cost_ang
        return get_coords(ang_from_parent)

    @staticmethod
    def plot_1d_clusters(cluster_ccf_file):
        """
        Creates directory of 1d cluster ccf plots for each sample
        Args:
            cluster_ccf_file: path to cluster_ccfs file

        """
        def scale_x(x):
            return 400 * x + 50

        patient = ''
        ccf_dist_keys = ['postDP_ccf_' + str(i / 100.) for i in range(101)]
        ccf_dict = {}
        with open(cluster_ccf_file, 'r') as clust_file:
            header = clust_file.readline().strip('\n\r').split('\t')
            for line in clust_file:
                fields = dict(zip(header, line.strip('\n\r').split('\t')))
                if not patient:
                    patient = fields['Patient_ID']
                c = int(fields['Cluster_ID'])
                sample = fields['Sample_ID']
                ccf_dist = [float(fields[k]) for k in ccf_dist_keys]
                ccf_dict.setdefault(sample, {})
                ccf_dict[sample][c] = ccf_dist
        if not os.path.exists('{}_1d_cluster_plots'.format(patient)):
            os.mkdir('{}_1d_cluster_plots'.format(patient))
        zero_str = functools.reduce(lambda acc, curr: acc + ' ' + str(scale_x(curr)) + ',365',
                                    np.arange(1., -0.01, -0.01), '')
        for sample in ccf_dict:
            document = dom.Document()
            max_bin = max(itertools.chain(*ccf_dict[sample].values()))
            y_max = math.ceil(max_bin * 20) / 20.
            if y_max - max_bin < 0.01 and y_max < 1.:
                y_max += 0.05

            def scale_y(y):
                return 365 - (345 * y / y_max)

            svg = document.appendChild(document.createElement('svg'))
            svg.setAttribute('baseProfile', 'full')
            svg.setAttribute('version', '1.1')
            svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
            svg.setAttribute('xmlns:ev', 'http://www.w3.org/2001/xml-events')
            svg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
            xaxis = svg.appendChild(document.createElement('g'))
            xaxis.setAttribute('transform', 'translate(0 365)')
            xaxis.setAttribute('fill', 'none')
            xaxis.setAttribute('font-size', '10')
            xaxis.setAttribute('font-family', 'sans-serif')
            xaxis.setAttribute('text-anchor', 'middle')
            xpath = xaxis.appendChild(document.createElement('path'))
            xpath.setAttribute('class', 'domain')
            xpath.setAttribute('stroke', 'currentColor')
            xpath.setAttribute('d', 'M50.5,6V0.5H450.5V6')
            for x in np.arange(0, 1.1, .1):
                x = round(x, 2)
                tick = xaxis.appendChild(document.createElement('g'))
                tick.setAttribute('class', 'tick')
                tick.setAttribute('opacity', '1')
                tick.setAttribute('transform', 'translate({})'.format(scale_x(x)))
                tickline = tick.appendChild(document.createElement('line'))
                tickline.setAttribute('stroke', 'currentColor')
                tickline.setAttribute('y2', '6')
                ticklabel = tick.appendChild(document.createElement('text'))
                ticklabel.setAttribute('fill', 'currentColor')
                ticklabel.setAttribute('y', '9')
                ticklabel.setAttribute('dy', '0.71em')
                ticklabel.appendChild(document.createTextNode(str(x)))
            yaxis = svg.appendChild(document.createElement('g'))
            yaxis.setAttribute('transform', 'translate(50 0)')
            yaxis.setAttribute('fill', 'none')
            yaxis.setAttribute('font-size', '10')
            yaxis.setAttribute('font-family', 'sans-serif')
            yaxis.setAttribute('text-anchor', 'end')
            ypath = yaxis.appendChild(document.createElement('path'))
            ypath.setAttribute('class', 'domain')
            ypath.setAttribute('stroke', 'currentColor')
            ypath.setAttribute('d', 'M-6,365.5H0.5V20.5H-6')
            for y in np.arange(0, y_max + 0.05, 0.05):
                y = round(y, 2)
                tick = yaxis.appendChild(document.createElement('g'))
                tick.setAttribute('class', 'tick')
                tick.setAttribute('opacity', '1')
                tick.setAttribute('transform', 'translate(0, {})'.format(scale_y(y)))
                tickline = tick.appendChild(document.createElement('line'))
                tickline.setAttribute('stroke', 'currentColor')
                tickline.setAttribute('x2', '-6')
                ticklabel = tick.appendChild(document.createElement('text'))
                ticklabel.setAttribute('fill', 'currentColor')
                ticklabel.setAttribute('x', '-9')
                ticklabel.setAttribute('dy', '0.32em')
                ticklabel.appendChild(document.createTextNode(str(y)))
            xlabel = svg.appendChild(document.createElement('text'))
            xlabel.setAttribute('x', '250')
            xlabel.setAttribute('y', '395')
            xlabel.setAttribute('font-size', '10px')
            xlabel.setAttribute('font-family', 'sans-serif')
            xlabel.setAttribute('text-anchor', 'middle')
            xlabel.appendChild(document.createTextNode('CCF'))
            ylabel = svg.appendChild(document.createElement('text'))
            ylabel.setAttribute('x', '10')
            ylabel.setAttribute('y', '192.5')
            ylabel.setAttribute('transform', 'rotate(-90 10 192.5)')
            ylabel.setAttribute('font-size', '10px')
            ylabel.setAttribute('font-family', 'sans-serif')
            ylabel.setAttribute('text-anchor', 'middle')
            ylabel.appendChild(document.createTextNode('Probability density'))
            for c in ccf_dict[sample]:
                dist_str = functools.reduce(
                    lambda acc, curr: acc + ' ' + str(scale_x(curr[0] * 0.01)) + ',' + str(scale_y(curr[1])),
                    enumerate(ccf_dict[sample][c]), '')
                dist = svg.appendChild(document.createElement('polygon'))
                dist.setAttribute('points', dist_str + zero_str)
                dist.setAttribute('fill', ClusterColors.get_rgb_string(c))
                dist.setAttribute('opacity', '0.5')
                dist.setAttribute('stroke', 'none')
            with open('{}_1d_cluster_plots/{}.cluster_ccfs.svg'.format(patient, sample), 'w') as f:
                svg.writexml(f)

    @staticmethod
    def plot_1d_mutations(mutation_ccf_file):
        """
        Creates directory of 1d mutation ccf plots for each sample
        Args:
            mutation_ccf_file: path to mut_ccfs file

        """
        def scale_x(x):
            return 400 * x + 50

        patient = ''
        ccf_dist_keys = ['preDP_ccf_' + str(i / 100.) for i in range(101)]
        ccf_dict = {}
        with open(mutation_ccf_file, 'r') as clust_file:
            header = clust_file.readline().strip('\n\r').split('\t')
            for line in clust_file:
                fields = dict(zip(header, line.strip('\n\r').split('\t')))
                if not patient:
                    patient = fields['Patient_ID']
                c = int(fields['Cluster_Assignment'])
                sample = fields['Sample_ID']
                ccf_dist = [float(fields[k]) for k in ccf_dist_keys]
                ccf_dict.setdefault(sample, {})
                ccf_dict[sample].setdefault(c, [])
                ccf_dict[sample][c].append(ccf_dist)
        if not os.path.exists('{}_1d_mutation_plots'.format(patient)):
            os.mkdir('{}_1d_mutation_plots'.format(patient))
        zero_str = functools.reduce(lambda acc, curr: acc + ' ' + str(scale_x(curr)) + ',365',
                                    np.arange(1., -0.01, -0.01), '')
        for sample in ccf_dict:
            document = dom.Document()
            max_bin = max(itertools.chain(*[itertools.chain(*d) for d in ccf_dict[sample].values()]))
            y_max = math.ceil(max_bin * 20) / 20.
            if y_max - max_bin < 0.01 and y_max < 1.:
                y_max += 0.05

            def scale_y(y):
                return 365 - (345 * y / y_max)

            svg = document.appendChild(document.createElement('svg'))
            svg.setAttribute('baseProfile', 'full')
            svg.setAttribute('version', '1.1')
            svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
            svg.setAttribute('xmlns:ev', 'http://www.w3.org/2001/xml-events')
            svg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
            xaxis = svg.appendChild(document.createElement('g'))
            xaxis.setAttribute('transform', 'translate(0 365)')
            xaxis.setAttribute('fill', 'none')
            xaxis.setAttribute('font-size', '10')
            xaxis.setAttribute('font-family', 'sans-serif')
            xaxis.setAttribute('text-anchor', 'middle')
            xpath = xaxis.appendChild(document.createElement('path'))
            xpath.setAttribute('class', 'domain')
            xpath.setAttribute('stroke', 'currentColor')
            xpath.setAttribute('d', 'M50.5,6V0.5H450.5V6')
            for x in np.arange(0, 1.1, .1):
                x = round(x, 2)
                tick = xaxis.appendChild(document.createElement('g'))
                tick.setAttribute('class', 'tick')
                tick.setAttribute('opacity', '1')
                tick.setAttribute('transform', 'translate({})'.format(scale_x(x)))
                tickline = tick.appendChild(document.createElement('line'))
                tickline.setAttribute('stroke', 'currentColor')
                tickline.setAttribute('y2', '6')
                ticklabel = tick.appendChild(document.createElement('text'))
                ticklabel.setAttribute('fill', 'currentColor')
                ticklabel.setAttribute('y', '9')
                ticklabel.setAttribute('dy', '0.71em')
                ticklabel.appendChild(document.createTextNode(str(x)))
            yaxis = svg.appendChild(document.createElement('g'))
            yaxis.setAttribute('transform', 'translate(50 0)')
            yaxis.setAttribute('fill', 'none')
            yaxis.setAttribute('font-size', '10')
            yaxis.setAttribute('font-family', 'sans-serif')
            yaxis.setAttribute('text-anchor', 'end')
            ypath = yaxis.appendChild(document.createElement('path'))
            ypath.setAttribute('class', 'domain')
            ypath.setAttribute('stroke', 'currentColor')
            ypath.setAttribute('d', 'M-6,365.5H0.5V20.5H-6')
            for y in np.arange(0, y_max + 0.05, 0.05):
                y = round(y, 2)
                tick = yaxis.appendChild(document.createElement('g'))
                tick.setAttribute('class', 'tick')
                tick.setAttribute('opacity', '1')
                tick.setAttribute('transform', 'translate(0, {})'.format(scale_y(y)))
                tickline = tick.appendChild(document.createElement('line'))
                tickline.setAttribute('stroke', 'currentColor')
                tickline.setAttribute('x2', '-6')
                ticklabel = tick.appendChild(document.createElement('text'))
                ticklabel.setAttribute('fill', 'currentColor')
                ticklabel.setAttribute('x', '-9')
                ticklabel.setAttribute('dy', '0.32em')
                ticklabel.appendChild(document.createTextNode(str(y)))
            xlabel = svg.appendChild(document.createElement('text'))
            xlabel.setAttribute('x', '250')
            xlabel.setAttribute('y', '395')
            xlabel.setAttribute('font-size', '10px')
            xlabel.setAttribute('font-family', 'sans-serif')
            xlabel.setAttribute('text-anchor', 'middle')
            xlabel.appendChild(document.createTextNode('CCF'))
            ylabel = svg.appendChild(document.createElement('text'))
            ylabel.setAttribute('x', '10')
            ylabel.setAttribute('y', '192.5')
            ylabel.setAttribute('transform', 'rotate(-90 10 192.5)')
            ylabel.setAttribute('font-size', '10px')
            ylabel.setAttribute('font-family', 'sans-serif')
            ylabel.setAttribute('text-anchor', 'middle')
            ylabel.appendChild(document.createTextNode('Probability density'))
            for c in ccf_dict[sample]:
                for hist in ccf_dict[sample][c]:
                    dist_str = functools.reduce(
                        lambda acc, curr: acc + ' ' + str(scale_x(curr[0] * 0.01)) + ',' + str(scale_y(curr[1])),
                        enumerate(hist), '')
                    dist = svg.appendChild(document.createElement('polygon'))
                    dist.setAttribute('points', dist_str + zero_str)
                    dist.setAttribute('fill', ClusterColors.get_rgb_string(c))
                    dist.setAttribute('opacity', '0.1')
                    dist.setAttribute('stroke', 'none')
            with open('{}_1d_mutation_plots/{}.mutations_ccfs.svg'.format(patient, sample), 'w') as f:
                svg.writexml(f)

    def write_pdf_report(self):
        """

        Returns:

        """
        raise NotImplementedError

    def write_maf(self):
        """

        Returns:

        """
        raise NotImplementedError

    def write_patient_cluster_ccfs(self, patient, cluster_ccfs, aliases=None):
        """
        Serialize cluster ccfs
        Args:
            patient: Patient instance
            cluster_ccfs: maps cluster ID to ccf histogram list (corresponding to samples in patient.sample_list)
            aliases: list of sample aliases (corresponding to samples in patient.sample_list)
        """
        header = ['Patient_ID', 'Sample_ID', 'Sample_Alias', 'Cluster_ID', 'postDP_ccf_mean', 'postDP_ccf_CI_low',
                  'postDP_ccf_CI_high']
        header.extend('postDP_ccf_{}'.format(float(x) / 100) for x in range(101))
        if aliases is None:
            aliases = ('',)*len(patient.sample_list)
        with open('{}.cluster_ccfs.txt'.format(patient.indiv_name), 'w') as f:
            f.write('\t'.join(header))
            for cluster in cluster_ccfs:
                for i, sample in enumerate(patient.sample_list):
                    mean, high, low = self._get_mean_high_low(np.array(cluster_ccfs[cluster][i]))
                    line = [patient.indiv_name, sample.sample_name, aliases[i], str(cluster), str(mean), str(low), str(high)]
                    line.extend(map(str, cluster_ccfs[cluster][i]))
                    f.write('\n' + '\t'.join(map(str, line)))

    def write_patient_mut_ccfs(self, patient, cluster_ccfs, aliases=None):
        """
        Serialize mutation ccfs
        Args:
            patient: Patient instance
            cluster_ccfs: maps cluster ID to ccf histogram list (corresponding to samples in patient.sample_list)
            aliases: list of sample aliases (corresponding to samples in patient.sample_list)
        """
        header = ['Patient_ID', 'Sample_ID', 'Sample_Alias', 'Hugo_Symbol', 'Chromosome', 'Start_position',
                  'Reference_Allele', 'Tumor_Seq_Allele', 't_ref_count', 't_alt_count', 'Protein_change',
                  'Variant_Classification', 'Variant_Type', 'Cluster_Assignment', 'Allelic_CN_minor',
                  'Allelic_CN_major', 'preDP_ccf_mean', 'preDP_ccf_CI_low', 'preDP_ccf_CI_high', 'clust_ccf_mean',
                  'clust_ccf_CI_low', 'clust_ccf_CI_high']
        header.extend('preDP_ccf_{}'.format(float(x) / 100) for x in range(101))
        if aliases is None:
            aliases = ('',)*len(patient.sample_list)
        with open('{}.mut_ccfs.txt'.format(patient.indiv_name), 'w') as f:
            f.write('\t'.join(header))
            for i, sample in enumerate(patient.sample_list):
                for mut in sample.concordant_variants:
                    if mut.type != 'CNV' and hasattr(mut, 'cluster_assignment') and mut.cluster_assignment is not None:
                        mut_mean, mut_high, mut_low = self._get_mean_high_low(np.array(mut.ccf_1d).astype(float))
                        c = mut.cluster_assignment
                        clust_mean, clust_high, clust_low = self._get_mean_high_low((np.array(cluster_ccfs[c][i])))
                        line = [patient.indiv_name, sample.sample_name, aliases[i], mut.gene if mut.gene else 'Unknown',
                                str(mut.chrN), str(mut.pos), mut.ref, mut.alt, str(mut.ref_cnt), str(mut.alt_cnt),
                                str(mut.prot_change), mut.mut_category, mut.type,
                                str(mut.cluster_assignment), str(mut.local_cn_a1), str(mut.local_cn_a2),
                                str(mut_mean), str(mut_low), str(mut_high), str(clust_mean), str(clust_low),
                                str(clust_high)]
                        line.extend(map(str, mut.ccf_1d))
                        f.write('\n' + '\t'.join(map(str, line)))

    def write_patient_cnvs(self, patient, cluster_ccfs):
        header = ['Patient_ID', 'Sample_ID', 'Sample_Alias', 'Event_Name', 'Chromosome', 'Start',
                  'End', 'Variant_Type', 'local_cn', 'Cluster_Assignment', 'preDP_ccf_mean',
                  'preDP_ccf_CI_low', 'preDP_ccf_CI_high', 'clust_ccf_mean', 'clust_ccf_CI_low', 'clust_ccf_CI_high']
        header.extend('preDP_ccf_{}'.format(float(x) / 100) for x in range(101))
        with open('{}.cnvs.txt'.format(patient.indiv_name), 'w') as f:
            f.write('\t'.join(header))
            for i, sample in enumerate(patient.sample_list):
                for mut in sample.low_coverage_mutations.values():
                    if mut.type == 'CNV':
                        mut_mean, mut_high, mut_low = self._get_mean_high_low(np.array(mut.ccf_1d))
                        c = mut.cluster_assignment if hasattr(mut, 'cluster_assignment') else None
                        if c:
                            clust_mean, clust_high, clust_low = self._get_mean_high_low((np.array(cluster_ccfs[c][i])))
                        else:
                            clust_mean, clust_high, clust_low = (None, None, None)
                        line = [patient.indiv_name, sample.sample_name, '', mut.event_name, mut.chrN, mut.start,
                                mut.end, 'CNV', mut.local_cn_a1 if mut.a1 else mut.local_cn_a2,
                                mut.cluster_assignment, mut_mean, mut_low, mut_high, clust_mean, clust_low, clust_high]
                        line.extend(mut.ccf_1d)
                        f.write('\n' + '\t'.join(map(str, line)))

    @staticmethod
    def _get_mean_high_low(ccf, confidence=.90):
        """
        Get ccf mean and confidence interval

        Args:
            ccf: CCF histogram

        Returns:
            Mean, confidence upper bound, confidence lower bound
        """
        grid_size = len(ccf)
        mean = round(sum(ccf * np.arange(float(grid_size)) / (grid_size-1)), 2)
        high = int(mean * (grid_size-1))
        low = int(mean * (grid_size-1))
        while np.sum(ccf[low:high + 1]) < confidence:
            if high == grid_size-1:
                low -= 1
            elif low == 0 or ccf[high + 1] > ccf[low - 1]:
                high += 1
            else:
                low -= 1
        return mean, float(high) / (grid_size-1), float(low) / (grid_size-1)

    def draw_tree_plot(self):
        """

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def write_tree_json(tree, indiv_id):
        """

        Returns:
        """
        nodes = list(tree.nodes.keys())
        edges = tree.edges
        root = tree.root.identifier
        with open(indiv_id + '.tree.json', 'w') as tree_fh:
            json.dump({'nodes': nodes, 'edges': edges, 'root': root}, tree_fh)

    """
    @staticmethod
    def write_tree_tsv(trees_counts, trees_ll, indiv_id):        
        # add header
        header = ['n_iter', 'log_lik', 'edges']
        with open(indiv_id + '_build_tree_posteriors.tsv', 'w') as writer:
            writer.write('\t'.join(header) + '\n')
            for i in range(len(trees_counts)):
                tree_edges, count = trees_counts[i]
                likelihood = str(trees_ll[i])
                line = [str(count), likelihood, str(tree_edges)]
                writer.write('\t'.join(line) + '\n')
    """

    @staticmethod
    def reformat_edges_for_output(edges):
        reformatted_edges = [str(edge[0]) + '-' + str(edge[1]) for edge in edges]
        reformatted_edges.append('None-1')
        return ','.join(reformatted_edges)

    @staticmethod
    def reformat_edges_for_input(edges):
        edges = [edge.split('-') for edge in edges.split(',')]
        reformatted_edges = []
        for (parent, child) in edges:
            try:
                parent = int(parent)
            except ValueError:
                parent = None
            try:
                child = int(child)
            except ValueError:
                child = None
            reformatted_edges.append((parent, child))
        return reformatted_edges

    def write_tree_tsv(self, trees_mcmc_trace, indiv_id):
        """
        Returns:
        """
        import pandas as pd
        df = pd.DataFrame(trees_mcmc_trace)
        df = df.sort_values(by='n_iter', ascending=False)
        df['edges'] = df['edges'].apply(self.reformat_edges_for_output)
        df = df[['n_iter', 'likelihood', 'edges']]
        output_file = indiv_id + '_build_tree_posteriors.tsv'
        df.to_csv(output_file, sep='\t', index=False)

    @staticmethod
    def write_all_cell_abundances(all_cell_abundances, indiv_id):
        header = ['Patient_ID', 'Sample_ID', 'Iteration', 'Cluster_ID', 'Abundance']
        with open(indiv_id + '_cell_population_mcmc_trace.tsv', 'w') as writer:
            writer.write('\t'.join(header) + '\n')
            for sample_id, sample_mcmc_trace in all_cell_abundances.items():
                for iteration, abundances in enumerate(sample_mcmc_trace):
                    for cluster, amount in abundances.items():
                        writer.write('\t'.join([indiv_id, sample_id, str(iteration), str(cluster), str(amount)]) + '\n')

    @staticmethod
    def write_cell_abundances_tsv(cell_abundances, cells_ancestry, indiv_id):
        """

        Returns:

        """
        # add header
        header = ['Patient_ID', 'Sample_ID', 'Cell_population', 'Cell_abundance']
        with open(indiv_id + '_cell_population_abundances.tsv', 'w') as writer:
            writer.write('\t'.join(header) + '\n')
            for (sample_id, abundances) in cell_abundances.items():
                for (cluster_id, abundance) in abundances.items():
                    population = '_'.join(['CL{}'.format(cl) for cl in cells_ancestry[cluster_id]])
                    line = [indiv_id, sample_id, population, str(abundance)]
                    writer.write('\t'.join(line) + '\n')

    @staticmethod
    def write_constrained_ccf_tsv(constrained_ccf_configuration, cells_ancestry, indiv_id):
        """

        Returns:

        """
        # add header
        header = ['Patient_ID', 'Sample_ID', 'Cell_population', 'Constrained_CCF']
        with open(indiv_id + '_constrained_ccf.tsv', 'w') as writer:
            writer.write('\t'.join(header) + '\n')
            for (sample_id, sample_config) in constrained_ccf_configuration.items():
                for (cluster_id, abundance) in sample_config:
                    population = '_'.join(['CL{}'.format(cl) for cl in cells_ancestry[cluster_id]])
                    line = [indiv_id, sample_id, population, str(abundance)]
                    writer.write('\t'.join(line) + '\n')

    def write_timing_tsv(self, timing_engine):
        """

        Returns:

        """
        header = ['Patient_ID', 'Event Name', 'Chromosome', 'Arm', 'Start_position', 'Reference_Allele',
                  'Tumor_Seq_Allele', 't_ref_count', 't_alt_count', 'Allelic_CN_minor', 'Allelic_CN_major',
                  'Allelic_CN', 'pi_mean', 'pi_low', 'pi_high']
        header.extend('pi_{}'.format(float(x) / 100) for x in range(101))
        patient_name = timing_engine.patient.indiv_name
        null_pi = ('',) * 101
        with open(patient_name + '.timing.tsv', 'w') as f:
            f.write('\t'.join(header))
            if timing_engine.WGD is not None:
                pi_mean, pi_high, pi_low = self._get_mean_high_low(timing_engine.WGD.pi_dist)
                line = [patient_name, 'WGD', '', '', '', '', '', '', '', '', '', '', pi_mean, pi_low, pi_high]
                line.extend(timing_engine.WGD.pi_dist)
                f.write('\n' + '\t'.join(map(str, line)))
            for cn_event_list in timing_engine.all_cn_events.values():
                for cn_event in cn_event_list:
                    if cn_event.pi_dist is not None and not any(np.isnan(cn_event.pi_dist)):
                        pi_mean, pi_high, pi_low = self._get_mean_high_low(cn_event.pi_dist)
                        pi_dist = cn_event.pi_dist
                    else:
                        pi_mean = pi_high = pi_low = ''
                        pi_dist = null_pi
                    line = [patient_name, cn_event.event_name, cn_event.chrN, cn_event.arm, '', '', '', '', '',
                            cn_event.cn_a1, cn_event.cn_a2, cn_event.allelic_cn, pi_mean, pi_low, pi_high]
                    line.extend(pi_dist)
                    f.write('\n' + '\t'.join(map(str, line)))
            for mut in timing_engine.mutations.values():
                if mut.pi_dist is not None and not any(np.isnan(mut.pi_dist)):
                    pi_mean, pi_high, pi_low = self._get_mean_high_low(mut.pi_dist)
                    pi_dist = mut.pi_dist
                else:
                    pi_mean = pi_high = pi_low = ''
                    pi_dist = null_pi
                line = [patient_name, mut.event_name, mut.chrN, '', mut.pos, mut.ref, mut.alt, mut.ref_cnt, mut.alt_cnt,
                        mut.local_cn_a1, mut.local_cn_a2, '', pi_mean, pi_low, pi_high]
                line.extend(pi_dist)
                f.write('\n' + '\t'.join(map(str, line)))
        if timing_engine.WGD is not None:
            wgd_header = ['Patient_ID', 'Event Name', 'Chromosome', 'Arm', 'Allelic_CN_minor', 'Allelic_CN_major',
                          'Allelic_CN', 'pi_mean', 'pi_low', 'pi_high']
            wgd_header.extend('pi_{}'.format(float(x) / 100) for x in range(101))
            with open(patient_name + '.WGD_supporting_events.timing.tsv', 'w') as f:
                f.write('\t'.join(wgd_header))
                for state in timing_engine.WGD.supporting_arm_states:
                    for cn_event in state.cn_events:
                        if cn_event.pi_dist is not None and not any(np.isnan(cn_event.pi_dist)):
                            pi_mean, pi_high, pi_low = self._get_mean_high_low(cn_event.pi_dist)
                            pi_dist = cn_event.pi_dist
                        else:
                            pi_mean = pi_high = pi_low = ''
                            pi_dist = null_pi
                        line = [patient_name, cn_event.event_name, cn_event.chrN, cn_event.arm,
                                cn_event.cn_a1, cn_event.cn_a2, cn_event.allelic_cn, pi_mean, pi_low, pi_high]
                        line.extend(pi_dist)
                        f.write('\n' + '\t'.join(map(str, line)))

    @staticmethod
    def write_comp_table(indiv_id, comps):
        with open(indiv_id + '.comp.tsv', 'w') as f:
            f.write('samp\teve1\teve2\tp1->2\tp2->1\tunknown')
            for eve_pair in comps:
                p1_2, p2_1, p_unknown = comps[eve_pair]
                eve1, eve2 = eve_pair
                f.write('\n' + '\t'.join(map(str, [indiv_id, eve1, eve2, p1_2, p2_1, p_unknown])))

    def draw_timing_graph(self, indiv_id, comps, coincidence_thresh=.8, edge_thresh=.5, eves_per_row=2,
                          nodes_per_layer=3, dist_between_layers=200, figsize=(8, 8)):
        """
        Write a png for the comparison graph
        Args:
            indiv_id: patient id
            comps: comparison dict from SinglePatientTiming.compare_events

        Returns: list of nodes, list of edges, dict of node positions

        """
        DG = self._get_timing_graph(comps, coincidence_thresh=coincidence_thresh, edge_thresh=edge_thresh,
                                    eves_per_row=eves_per_row)
        nodes = list(DG.nodes)
        edges = list(DG.edges)
        pos = self._get_timing_graph_coordinates(nodes, edges, nodes_per_layer=nodes_per_layer,
                                                 dist_between_layers=dist_between_layers)
        mpl_pos = {node: (pos[node][1], -pos[node][0]) for node in nodes}
        plt.figure(figsize=figsize)
        ax = plt.gca()
        plt.axis('off')
        nx.draw_networkx(DG, mpl_pos, ax=ax, node_size=500, font_size=10, node_color='orange')
        xmin, xmax = ax.get_xlim()
        xmean = (xmin + xmax) / 2
        xwidth = xmax - xmean
        ymin, ymax = ax.get_ylim()
        ymean = (ymin + ymax) / 2
        ywidth = ymax - ymean
        ax.set_xlim(xmean - xwidth * 1.2, xmean + xwidth * 1.2)
        ax.set_ylim(ymean - ywidth * 1.2, ymean + ywidth * 1.2)
        ax.set_title(indiv_id + ' timing graph')
        plt.savefig(indiv_id + '.comp_graph.png')
        return nodes, edges, pos

    @staticmethod
    def _get_timing_graph(comps, coincidence_thresh=.8, edge_thresh=.5, eves_per_row=2):
        """
        Get nodes and edges from a comparison table
        Args:
            comps: comparison table
            coincidence_thresh: probability threshold above which to merge events into a single node
            edge_thresh: probability threshold above which to create a directed edge
            eves_per_row: number of events per row in the label

        Returns:
        networkx DiGraph object
        """
        all_events = set(itertools.chain(*comps))
        neighbors = {eve: set() for eve in all_events}
        for (eve1, eve2), (p1_2, p2_1, p_unknown) in comps.items():
            if p_unknown > coincidence_thresh:
                neighbors[eve1].add(eve2)
                neighbors[eve2].add(eve1)

        def _BK(P, neighbors, R=frozenset(), X=frozenset()):
            if not P and not X:
                yield R
            else:
                for v in P:
                    for r in _BK(P & neighbors[v], neighbors, R=R | {v}, X=X & neighbors[v]):
                        yield r
                    P = P - {v}
                    X = X | {v}

        nodes = list(_BK(all_events, neighbors))
        DG = nx.DiGraph()
        labels = []
        for node in nodes:
            label = ''
            for i, eve in enumerate(node):
                if i == 0:
                    label += eve
                elif i % eves_per_row == 0:
                    label += '\n' + eve
                else:
                    label += ', ' + eve
            labels.append(label)
            DG.add_node(label)
        for i1, i2 in itertools.combinations(range(len(nodes)), 2):
            label1 = labels[i1]
            eve1 = next(iter(nodes[i1]))
            label2 = labels[i2]
            eve2 = next(iter(nodes[i2]))
            if (eve1, eve2) in comps:
                p1_2, p2_1, p_unknown = comps[(eve1, eve2)]
            else:
                p2_1, p1_2, p_unknown = comps[(eve2, eve1)]
            if p1_2 > edge_thresh:
                DG.add_edge(label1, label2)
            elif p2_1 > edge_thresh:
                DG.add_edge(label2, label1)
        for edge in list(DG.edges):
            if len(list(nx.all_simple_paths(DG, *edge))) > 1:
                DG.remove_edge(*edge)
        return DG

    @staticmethod
    def _get_timing_graph_coordinates(nodes, edges, nodes_per_layer=3, dist_between_layers=200, layer_width=400):
        """
        Coffman-Graham algorithm for graph drawing
        Args:
            nodes: list of nodes
            edges: list of edges
            nodes_per_layer: nodes per layer

        Returns:
        dict of positions for each node
        """
        parent_dict = {node: [] for node in nodes}
        child_dict = {node: [] for node in nodes}
        for edge in edges:
            parent_dict[edge[1]].append(edge[0])
            child_dict[edge[0]].append(edge[1])
        ordered_nodes = []
        nodes_to_order = []
        for node in nodes:
            if parent_dict[node]:
                nodes_to_order.append(node)
            else:
                ordered_nodes.append(node)

        def order_nodes(node):
            parents = parent_dict[node]
            for parent in parents:
                if parent not in ordered_nodes:
                    return [-1]
            order = []
            for idx, ordered_node in enumerate(reversed(ordered_nodes)):
                if ordered_node in parents:
                    order.append(idx)
            return order

        while nodes_to_order:
            node_idx = max(range(len(nodes_to_order)), key=lambda idx: order_nodes(nodes_to_order[idx]))
            ordered_nodes.append(nodes_to_order.pop(node_idx))

        node_layers = dict.fromkeys(nodes, 0)
        nodes_by_layer = {}
        for node in reversed(ordered_nodes):
            layer = max(node_layers[child] for child in child_dict[node]) + 1 if child_dict[node] else 0
            nodes_by_layer.setdefault(layer, [])
            while len(nodes_by_layer[layer]) == nodes_per_layer:
                layer += 1
                nodes_by_layer.setdefault(layer, [])
            node_layers[node] = layer
            nodes_by_layer[layer].append(node)

        node_positions = {}
        for layer in nodes_by_layer:
            nodes_on_layer = nodes_by_layer[layer]
            n_nodes = len(nodes_on_layer)
            for i, node in enumerate(nodes_on_layer):
                x = -dist_between_layers * layer
                y = (i - (n_nodes - 1) / 2.) * layer_width / (nodes_per_layer - 1)
                node_positions[node] = (x, y)

        return node_positions

    def generate_html_from_timing(self, indiv_id, timing_engine, comps, drivers=()):
        nodes, edges, pos = self.draw_timing_graph(indiv_id, comps)
        eve_list = []
        cnv_pi_dists = {}
        driver_pi_dists = {}
        if timing_engine.WGD is not None:
            pi_mean, pi_high, pi_low = self._get_mean_high_low(timing_engine.WGD.pi_dist)
            pi_score = '{} ({}-{})'.format(pi_mean, pi_low, pi_high)
            eve_list.append({'Event name': 'WGD', 'Chromosome': '', 'Position': '', 'Pi score': pi_score})
            cnv_pi_dists['WGD'] = list(timing_engine.WGD.pi_dist)
        for eve in itertools.chain(*timing_engine.all_cn_events.values()):
            pi_mean, pi_high, pi_low = self._get_mean_high_low(eve.pi_dist)
            pi_score = '{} ({}-{})'.format(pi_mean, pi_low, pi_high)
            eve_list.append({'Event name': eve.event_name, 'Chromosome': eve.chrN, 'Position': '',
                             'Pi score': pi_score})
            cnv_pi_dists[eve.event_name] = list(eve.pi_dist)
        for mut in timing_engine.mutations.values():
            pi_mean, pi_high, pi_low = self._get_mean_high_low(mut.pi_dist)
            pi_score = '{} ({}-{})'.format(pi_mean, pi_low, pi_high)
            eve_list.append({'Event name': mut.event_name, 'Chromosome': mut.chrN, 'Position': mut.pos,
                             'Pi score': pi_score})
            if mut.event_name.split('_')[0] in drivers:
                driver_pi_dists[mut.event_name] = list(mut.pi_dist)
        with open(os.path.dirname(__file__) + '/timing_report_template.html_', 'r') as fh_in, open(
                indiv_id + '.phylogic_timing_report.html', 'w') as fh_out:
            html_template = fh_in.read()
            fh_out.write(HTMLTemplate(html_template).substitute(**{
                'indiv_id': indiv_id,
                'comp_graph_options': json.dumps({'nodes': nodes, 'edges': edges, 'pos': pos}),
                'boxplot_options': json.dumps({'pi_distributions': cnv_pi_dists}),
                'cnv_pi_histograms_options': json.dumps({'pi_distributions': cnv_pi_dists}),
                'driver_pi_histograms_options': json.dumps({'pi_distributions': driver_pi_dists}),
                'timing_table_options': json.dumps({'eve_list': eve_list})
            }))


class ClusterColors(object):
    # Cluster colors
    color_list = [[166, 17, 129],
                  [39, 140, 24],
                  [103, 200, 243],
                  [248, 139, 16],
                  [16, 49, 41],
                  [93, 119, 254],
                  [152, 22, 26],
                  [104, 236, 172],
                  [249, 142, 135],
                  [55, 18, 48],
                  [83, 82, 22],
                  [247, 36, 36],
                  [0, 79, 114],
                  [243, 65, 132],
                  [60, 185, 179],
                  [185, 177, 243],
                  [139, 34, 67],
                  [178, 41, 186],
                  [58, 146, 231],
                  [130, 159, 21],
                  [161, 91, 243],
                  [131, 61, 17],
                  [248, 75, 81],
                  [32, 75, 32],
                  [45, 109, 116],
                  [255, 169, 199],
                  [55, 179, 113],
                  [34, 42, 3],
                  [56, 121, 166],
                  [172, 60, 15],
                  [115, 76, 204],
                  [21, 61, 73],
                  [67, 21, 74],  # Additional colors, uglier and bad
                  [123, 88, 112],
                  [87, 106, 46],
                  [37, 66, 58],
                  [132, 79, 62],
                  [71, 58, 32],
                  [59, 104, 114],
                  [46, 107, 90],
                  [84, 68, 73],
                  [90, 97, 124],
                  [121, 66, 76],
                  [104, 93, 48],
                  [49, 67, 82],
                  [71, 95, 65],
                  [127, 85, 44],  # even more additional colors, gray
                  [88, 79, 92],
                  [220, 212, 194],
                  [35, 34, 36],
                  [200, 220, 224],
                  [73, 81, 69],
                  [224, 199, 206],
                  [120, 127, 113],
                  [142, 148, 166],
                  [153, 167, 156],
                  [162, 139, 145],
                  [0, 0, 0]]  # black

    @classmethod
    def get_rgb_string(cls, c):
        return 'rgb({},{},{})'.format(*cls.color_list[c])

    @classmethod
    def get_hex_string(cls, c):
        return '#{:02X}{:02X}{:02X}'.format(*cls.color_list[c])


class HTMLTemplate(Template):
    delimiter = '@@'

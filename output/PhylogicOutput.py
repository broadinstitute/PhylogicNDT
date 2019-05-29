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

import matplotlib

# turn off the display var for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        if len(np.unique(time_points)) == 1:
            time_points = []
        cluster_dict = {c: {'muts': {}} for c in range(1, len(results.clust_CCF_dens) + 1)}
        n_muts = dict.fromkeys(range(1, len(results.clust_CCF_dens) + 1), 0)
        c_drivers = {c: [] for c in range(1, len(results.clust_CCF_dens) + 1)}
        for i, sample in enumerate(patient.sample_list):
            for mut in sample.mutations:
                if hasattr(mut, 'cluster_assignment'):
                    c = mut.cluster_assignment
                    if mut.gene and mut.prot_change:
                        mut_name = '_'.join((mut.gene, mut.prot_change))
                    elif mut.gene:
                        mut_name = ':'.join((mut.gene, str(mut.pos), mut.ref, mut.alt))
                    else:
                        mut_name = ':'.join((str(mut.chrN), str(mut.pos), mut.ref, mut.alt))
                    ccf_hat, ccf_high, ccf_low = self._get_mean_high_low(mut.ccf_1d)
                    alt_cnt = mut.alt_cnt
                    ref_cnt = mut.ref_cnt
                    cluster_dict[c]['muts'].setdefault(mut_name, {'ccf_hat': [], 'alt_cnt': [], 'ref_cnt': []})
                    cluster_dict[c]['muts'][mut_name]['ccf_hat'].append(ccf_hat)
                    cluster_dict[c]['muts'][mut_name]['alt_cnt'].append(alt_cnt)
                    cluster_dict[c]['muts'][mut_name]['ref_cnt'].append(ref_cnt)
                    if len(sample_names) == 1:
                        cluster_dict[c]['muts'][mut_name]['ccf_dist'] = list(map(float, mut.ccf_1d))
                    if i == 0:
                        n_muts[c] += 1
                        if mut_name.split('_')[0] in drivers:
                            c_drivers[c].append(mut_name)
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
                header = f.readline().strip().split('\t')
                tx_idx = header.index('tx')
                tx_start_idx = header.index('tx_start')
                tx_end_idx = header.index('tx_end')
                for line in f:
                    fields = line.strip().split('\t')
                    treatment_data.append({'tx': fields[tx_idx],
                                           'tx_start': float(fields[tx_start_idx]),
                                           'tx_end': float(fields[tx_end_idx])})
        else:
            treatment_data = None

        self.write_html_report(patient_id, sample_names, cluster_dict, time_points=time_points,
                               treatment_data=treatment_data)

    def generate_html_from_tree(self, mutation_ccf_file, cluster_ccf_file, tree, abundances, sif=None, drivers=(),
                                treatment_file=None):
        sample_names = []
        aliases = []
        timepoints = []
        tumor_sizes = []
        treatment_data = None
        if sif:
            with open(sif, 'r') as sif_file:
                header = sif_file.readline().strip().lower().split('\t')
                sample_idx = header.index('sample') if 'sample' in header else header.index('sample_id')
                alias_idx = header.index('alias') if 'alias' in header else None
                timepoint_idx = header.index('timepoint') if 'timepoint' in header else None
                tumor_size_idx = header.index('tumor_size') if 'tumor_size' in header else None
                for line in sif_file:
                    fields = line.strip().split('\t')
                    sample_names.append(fields[sample_idx])
                    if alias_idx is not None:
                        aliases.append(fields[alias_idx])
                    if timepoint_idx is not None:
                        timepoints.append(float(fields[timepoint_idx]))
                    if tumor_size_idx is not None:
                        tumor_sizes.append(float(fields[tumor_size_idx]))
            if len(timepoints) == len(sample_names):
                sorting_idx = sorted(range(len(sample_names)), key=lambda k: timepoints[k])
                sample_names = [sample_names[i] for i in sorting_idx]
                timepoints = [timepoints[i] for i in sorting_idx]
                if len(tumor_sizes) == len(sample_names):
                    tumor_sizes = [tumor_sizes[i] for i in sorting_idx]
                if len(aliases) == len(sample_names):
                    aliases = [aliases[i] for i in sorting_idx]
                if treatment_file:
                    treatment_data = []
                    with open(treatment_file, 'r') as f:
                        header = f.readline().strip().split('\t')
                        tx_idx = header.index('tx')
                        tx_start_idx = header.index('tx_start')
                        tx_end_idx = header.index('tx_end')
                        for line in f:
                            fields = line.strip().split('\t')
                            treatment_data.append({'tx': fields[tx_idx],
                                                   'tx_start': float(fields[tx_start_idx]),
                                                   'tx_end': float(fields[tx_end_idx])})
            else:
                timepoints = list(range(len(sample_names)))
            if len(tumor_sizes) != len(sample_names):
                tumor_sizes = [1.] * len(sample_names)
            if len(aliases) != len(sample_names):
                aliases = []

        cluster_dict = {}
        cluster_ccfs = {}
        patient = ''
        with open(cluster_ccf_file, 'r') as clust_file:
            header = clust_file.readline().strip().split('\t')
            patient_idx = header.index('Patient_ID')
            cluster_idx = header.index('Cluster_ID')
            sample_idx = header.index('Sample_ID')
            alias_idx = header.index('Sample_Alias')
            ccf_hat_idx = header.index('postDP_ccf_mean')
            ccf_high_idx = header.index('postDP_ccf_CI_high')
            ccf_low_idx = header.index('postDP_ccf_CI_low')
            ccf_dist_idx = header.index('postDP_ccf_0.0')
            for line in clust_file:
                fields = line.strip().split('\t')
                if not patient:
                    patient = fields[patient_idx]
                c = int(fields[cluster_idx])
                sample = fields[sample_idx]
                alias = fields[alias_idx]
                ccf_hat = float(fields[ccf_hat_idx])
                ccf_high = float(fields[ccf_high_idx])
                ccf_low = float(fields[ccf_low_idx])
                cluster_dict.setdefault(c, {'ccf_hat': [], 'ccf_high': [], 'ccf_low': [], 'muts': {},
                                            'drivers': [], 'tumor_abundance': []})
                cluster_dict[c]['ccf_hat'].append(ccf_hat)
                cluster_dict[c]['ccf_high'].append(ccf_high)
                cluster_dict[c]['ccf_low'].append(ccf_low)
                if sample not in sample_names:
                    sample_names.append(sample)
                    if alias:
                        aliases.append(alias)
                ccf_dist = list(map(float, fields[ccf_dist_idx:ccf_dist_idx + 101]))
                cluster_ccfs.setdefault(sample, {})
                cluster_ccfs[sample][c] = np.array(ccf_dist)
                if len(sample_names) == 1:
                    cluster_dict[c]['ccf_dist'] = ccf_dist
        n_muts = dict.fromkeys(cluster_dict, 0)
        with open(mutation_ccf_file, 'r') as mut_file:
            header = mut_file.readline().strip().split('\t')
            sample_idx = header.index('Sample_ID')
            hugo_idx = header.index('Hugo_Symbol')
            prot_change_idx = header.index('Protein_change')
            chrom_idx = header.index('Chromosome')
            pos_idx = header.index('Start_position')
            ref_idx = header.index('Reference_Allele')
            alt_idx = header.index('Tumor_Seq_Allele')
            ref_cnt_idx = header.index('t_ref_count')
            alt_cnt_idx = header.index('t_alt_count')
            cluster_idx = header.index('Cluster_Assignment')
            ccf_hat_idx = header.index('preDP_ccf_mean')
            ccf_dist_idx = header.index('preDP_ccf_0.0')
            for line in mut_file:
                fields = line.strip().split('\t')
                sample = fields[sample_idx]
                hugo = fields[hugo_idx]
                prot_change = fields[prot_change_idx]
                chrom = fields[chrom_idx]
                pos = fields[pos_idx]
                ref = fields[ref_idx]
                alt = fields[alt_idx]
                ref_cnt = fields[ref_cnt_idx]
                alt_cnt = fields[alt_cnt_idx]
                c = int(fields[cluster_idx])
                ccf_hat = float(fields[ccf_hat_idx])
                if hugo != 'Unknown' and prot_change:
                    mut_name = '_'.join((hugo, prot_change))
                elif hugo != 'Unknown':
                    mut_name = ':'.join((hugo, pos, ref, alt))
                else:
                    mut_name = ':'.join((chrom, pos, ref, alt))
                cluster_dict[c]['muts'].setdefault(mut_name, {'ccf_hat': [], 'alt_cnt': [], 'ref_cnt': []})
                cluster_dict[c]['muts'][mut_name]['ccf_hat'].append(ccf_hat)
                cluster_dict[c]['muts'][mut_name]['alt_cnt'].append(alt_cnt)
                cluster_dict[c]['muts'][mut_name]['ref_cnt'].append(ref_cnt)
                if len(sample_names) == 1:
                    cluster_dict[c]['muts'][mut_name]['ccf_dist'] = list(map(float, fields[ccf_dist_idx:ccf_dist_idx + 101]))
                if sample == sample_names[0]:
                    n_muts[c] += 1
                    if hugo in drivers:
                        cluster_dict[c]['drivers'].append(mut_name)
        for c in cluster_dict:
            cluster_dict[c]['line_width'] = min(12, math.ceil(3 * n_muts[c] / 7.))
            cluster_dict[c]['color'] = ClusterColors.get_rgb_string(c)
        with open(tree, 'r') as tree_fh:
            header = tree_fh.readline().strip().split('\t')
            edges = eval(tree_fh.readline().strip().split('\t')[header.index('edges')])
        tree = BuildTree.Tree.Tree()
        for identifier in set(itertools.chain(*edges)):
            tree.add_node(identifier)
        for parent, child in edges:
            tree.add_edge(tree.get_node_by_id(parent), tree.get_node_by_id(child))
        tree.set_root(tree.get_node_by_id(1))
        child_dict = {None: [tree.root.identifier]}
        for c, node in tree.nodes.items():
            child_dict[c] = node.children
        pie_plots = []
        with open(abundances, 'r') as abun_fh:
            header = abun_fh.readline().strip().split('\t')
            sample_name_idx = header.index('Sample_ID')
            cell_population_idx = header.index('Cell_population')
            cell_abundance_idx = header.index('Constrained_CCF')
            sample_name = ''
            cell_abundances = {}
            for line in abun_fh:
                fields = line.strip().split('\t')
                c = int(re.search(r'(\d{1,2})$', fields[cell_population_idx]).group(1))
                cell_abundance = float(fields[cell_abundance_idx])
                cluster_dict[c]['tumor_abundance'].append(cell_abundance / 100.)
                if fields[sample_name_idx] != sample_name:
                    if sample_name:
                        pie_plots.append(self.make_pie_plot(tree, cell_abundances))
                    sample_name = fields[sample_name_idx]
                    cell_abundances = {c: cell_abundance}
                else:
                    cell_abundances[c] = cell_abundance
            pie_plots.append(self.make_pie_plot(tree, cell_abundances))

        self.write_html_report(patient, sample_names, cluster_dict, aliases=aliases, time_points=timepoints,
                               pie_plots=pie_plots, child_dict=child_dict, tumor_sizes=tumor_sizes,
                               treatment_data=treatment_data)

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

    @staticmethod
    def write_html_report(patient, sample_names, cluster_dict, aliases=None, time_points=None, pie_plots=None,
                          child_dict=None, treatment_data=None, tumor_sizes=None):
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
            child_dict: Dict mapping parent to list of children (with None mapping to [trunk])
            treatment_data: List of treatments of the format {"tx": str, "tx_start": num, "tx_end": num}
            tumor_sizes: List of tumor sizes of each sample

        Returns:
            None

        """
        sample_name_table_options = {'sample_names': sample_names}
        time_switch_options = {'show': False}
        cluster_switches_options = {'xlabels': sample_names, 'cluster_dict': cluster_dict}
        clustering_plot_options = {'xlabels': sample_names, 'cluster_dict': cluster_dict}
        tree_options = False
        treatment_plot_options = False
        mutation_visual_options = {'aliases': sample_names, 'cluster_dict': cluster_dict}
        cluster_table_options = {'aliases': sample_names, 'cluster_dict': cluster_dict}
        mutation_plot_options = {'xlabels': sample_names, 'cluster_dict': cluster_dict}
        fish_plot_options = False

        if time_points and child_dict and tumor_sizes:
            fish_plot_options = {'xlabels': sample_names, 'time_points': time_points,
                                 'tumor_sizes': tumor_sizes, 'cluster_dict': cluster_dict,
                                 'child_dict': child_dict}

        columns = ['Sample Name']
        if aliases:
            columns.append('Alias')
            sample_name_table_options['sample_aliases'] = aliases
            clustering_plot_options['xlabels'] = aliases
            cluster_switches_options['xlabels'] = aliases
            mutation_visual_options['aliases'] = aliases
            cluster_table_options['aliases'] = aliases
            mutation_plot_options['xlabels'] = aliases
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

        if child_dict:
            n_muts = dict.fromkeys(child_dict, 0)
            dist_from_parent = dict.fromkeys(itertools.chain(*child_dict.values()))
            for parent in child_dict:
                if parent is not None:
                    n_muts[parent] += len(cluster_dict[parent]['muts'])
                for child in child_dict[parent]:
                    if parent is None:
                        dist_from_parent[child] = 1.
                    else:
                        n_muts[child] += len(cluster_dict[parent]['muts'])
                        n_muts_parent = n_muts[parent]
                        diff_muts_child = len(cluster_dict[child]['muts'])
                        dist_from_parent[child] = float(diff_muts_child) / (n_muts_parent + diff_muts_child)
            print dist_from_parent
            tree_options = {'patient': patient,'cluster_dict': cluster_dict,
                            'child_dict': child_dict, 'dist_from_parent': dist_from_parent}

        if treatment_data:
            treatment_plot_options = {'treatments': treatment_data,
                                 'samples': time_points,
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
                'treatment_plot_options': json.dumps(treatment_plot_options),
                'mutation_visual_options': json.dumps(mutation_visual_options),
                'cluster_table_options': json.dumps(cluster_table_options),
                'mutation_plot_options': json.dumps(mutation_plot_options),
                'fish_plot_options': json.dumps(fish_plot_options)}))

    @staticmethod
    def make_pie_plot(tree, cluster_abundances):
        """
        Makes clonal abundance pie plot
        Args:
            tree: Tree class
            cluster_abundances: dict mapping nodes to sampled abundances

        Returns:
            Base-64 encoded png of the pie plot
        """
        plt.figure(figsize=(1, 1))
        ax = plt.gca()
        leftovers = {}
        node_order = list(tree.traverse_by_branch())
        for level, clusters in enumerate(tree.traverse_by_level()):
            pie_slices = dict(leftovers)
            for node in clusters:
                pie_slices[node] = cluster_abundances[node.identifier]
                sum_children = sum(cluster_abundances[c] for c in node.children)
                if sum_children < cluster_abundances[node.identifier]:
                    leftovers[node] = cluster_abundances[node.identifier] - sum_children
            x = []
            colors = []
            for node in node_order:
                if node in pie_slices:
                    x.append(pie_slices[node])
                    colors.append(ClusterColors.get_hex_string(node.identifier))
            ax.pie(x, colors=colors, radius=.9-(.1*level), wedgeprops={'edgecolor':'white','linewidth':'1'})
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        bytes_io = BytesIO()
        plt.savefig(bytes_io, bbox_inches='tight', pad_inches=0., transparent=True)
        plt.close()
        return base64.b64encode(bytes_io.getvalue()).decode('UTF-8')

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
                for mut in sample.mutations:
                    if hasattr(mut, 'cluster_assignment'):
                        mut_mean, mut_high, mut_low = self._get_mean_high_low(np.array(mut.ccf_1d))
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

    @staticmethod
    def write_tree_tsv(trees_counts, trees_ll, indiv_id):
        """

        Returns:

        """
        # add header
        header = ['n_iter', 'log_lik', 'edges']
        with open(indiv_id + '_build_tree_posteriors.tsv', 'w') as writer:
            writer.write('\t'.join(header) + '\n')
            for (tree_edges, count) in trees_counts:
                line = [str(count), '0.0', str(tree_edges)]
                writer.write('\t'.join(line) + '\n')

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
    def write_constrained_ccf_tsv(constrained_ccf, cells_ancestry, indiv_id):
        """

        Returns:

        """
        # add header
        header = ['Patient_ID', 'Sample_ID', 'Cell_population', 'Constrained_CCF']
        with open(indiv_id + '_constrained_ccf.tsv', 'w') as writer:
            writer.write('\t'.join(header) + '\n')
            for (sample_id, constrained_ccf) in constrained_ccf.items():
                for (cluster_id, abundance) in constrained_ccf.items():
                    population = '_'.join(['CL{}'.format(cl) for cl in cells_ancestry[cluster_id]])
                    line = [indiv_id, sample_id, population, str(abundance)]
                    writer.write('\t'.join(line) + '\n')

    def draw_timing_plot(self):
        """

        Returns:

        """
        raise NotImplementedError

    def write_timing_tsv(self):
        """

        Returns:

        """
        raise NotImplementedError


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

    @ classmethod
    def get_rgb_string(cls, c):
        return 'rgb({},{},{})'.format(*cls.color_list[c])

    @classmethod
    def get_hex_string(cls, c):
        return '#' + ''.join(
            ('0' + hex(rgb)[2:].upper() if rgb < 16 else hex(rgb)[2:].upper() for rgb in cls.color_list[c]))


class HTMLTemplate(Template):
    delimiter = '!!'

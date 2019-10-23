import sys, os

import operator
import numpy as np
import pandas as pd
import logging


from scipy.optimize import minimize
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors, rcParams


def run_tool(sif_file, abundances_file):
    parse_sif_file(sif_file)


def parse_sif_file(sif_file, mutation_ccf_file, patient_data):
    with open(sif_file, 'r') as reader:
        for line in reader:
            if not line.strip() == "":
                # for now, assume input file order is of the type sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint
                values = line.strip('\n').split('\t')
                if line.startswith('sample_id'):
                    header = line.strip('\n').split('\t')
                    header = {x: i for i, x in enumerate(header)}
                else:
                    sample_id = values[header['sample_id']]
                    seg_fn = values[header['seg_fn']]
                    purity = float(values[header['purity']])
                    timepoint = float(values[header['timepoint']])
                    logging.debug("Adding sample {}".format(sample_id))

                    patient_data.addSample(mutation_ccf_file, sample_id,
                                           input_type="post-clustering",
                                           timepoint_value=timepoint,
                                           seg_file=seg_fn,
                                           purity=purity)



abund_file = '/Users/lelagina/Broad/KIPPS/data/abundances/CRC-0001_cell_population_abundances.txt'
time_points = range(4)
cell_abundance = pd.read_csv(abund_file, sep='\t')
cell_abundance.head()

# Load TPM values
tpm_values = pd.read_csv('../data/TPM_per_patient/1.TPM.08052018.txt', sep='\t')

print len(tpm_values)

non_zero_tpm = tpm_values[
    (tpm_values['Sample_1'] != 0.00) & (tpm_values['Sample_2'] != 0.00) & (tpm_values['Sample_3'] != 0.00) & (
            tpm_values['Sample_4'] != 0.00)]
print len(non_zero_tpm)
diff = dict()
for index, row in non_zero_tpm.iterrows():
    gene_diff = abs(row['Sample_1'] - row['Sample_2']) + abs(row['Sample_2'] - row['Sample_3']) + abs(
        row['Sample_3'] - row['Sample_4'])
    diff[row['gene_name']] = gene_diff
sorted_diff = sorted(diff.items(), key=operator.itemgetter(1), reverse=True)
sorted_diff
genes_to_consider = list()
for (gene, diff_exp) in sorted_diff:
    if not gene.startswith('MT-'):
        genes_to_consider.append(gene)

cancer_genes = list()
with open('/Users/lelagina/Broad/KIPPS/data/Driver_genes_v1.0.txt', 'r') as reader:
    for line in reader:
        cancer_genes.append(line.strip('\n'))

chr17_genes = list()
with open('/Users/lelagina/Broad/KIPPS/data/gene_info/chrom17p_genes_gencode_v19.txt', 'r') as reader:
    for line in reader:
        chr17_genes.append(line.strip('\n'))

genes_to_plot = list()
for g in chr17_genes:
    if g in cancer_genes:
        genes_to_plot.append(g)
print genes_to_plot


def sample_abundances(cell_abundance):
    abundances = np.zeros((4, 5))
    for t in time_points:
        sample = cell_abundance[cell_abundance['Sample'] == t].sample(n=1)
        for i in range(1, 5):
            abundances[t, i - 1] = sample[str(i)]
        abundances[t, 4] = sample['Normal']
    return abundances / 100.0


def matrix_diff(X, Y):
    return np.linalg.norm(X - Y)


def compute_solution(exp_values, abundances, method='L-BFGS-B'):
    n = abundances.shape[1]
    fun = lambda x: matrix_diff(np.dot(abundances, x), exp_values)
    sol = minimize(fun, np.ones(n), method=method, bounds=[(0.01, None) for x in xrange(n)])
    return sol.x


def get_gene_tpm_values(gene_name):
    gene_tpm_values = list()
    for index, row in tpm_values.iterrows():
        if row['gene_name'] == gene_name:
            gene_tpm_values.append(row['Sample_1'])
            gene_tpm_values.append(row['Sample_2'])
            gene_tpm_values.append(row['Sample_3'])
            gene_tpm_values.append(row['Sample_4'])

    return np.array(gene_tpm_values)


def sample_gausian_noise(mean, std, num_samples):
    return np.random.normal(mean, std, num_samples)


def correct_for_purity():
    pass


cancer_genes = cancer_genes[:5]
iterations = 5
data = list()
values = set()

# genes_to_plot
for i in range(iterations):
    print i
    abundance = sample_abundances(cell_abundance)

    for gene in genes_to_plot[10:15]:
        gene_tpm = get_gene_tpm_values(gene)

        solution = compute_solution(gene_tpm, abundance)

        for e in gene_tpm:
            values.add((sum(solution) * 100, e))
        # print np.dot(abundance, solution) - gene_tpm
        # print 'Error ', sum(np.dot(abundance, solution) - gene_tpm)
        for j in range(len(solution)):
            data.append(dict(
                gene_name=gene,
                cluster=j,
                expression=solution[j]
            ))

df = pd.DataFrame(data)
df.head()


def main():
    sif_file = '/Users/lelagina/PycharmProjects/PhylogicNDT-BuildTree_rework/kipps/CRC-0001.sif'
    abundance_file = '/Users/lelagina/PycharmProjects/PhylogicNDT-BuildTree_rework/kipps/CRC-0001_cell_population_abundances.tsv'
    run_tool(sif_file, abundance_file)


if __name__ == "__main__":
    main()

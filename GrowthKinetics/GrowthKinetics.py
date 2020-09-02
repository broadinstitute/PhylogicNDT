import logging
from collections import defaultdict
import numpy as np


def run_tool(args):
    logging.debug('Arguments {}'.format(args))
    import data.Patient as Patient
    from .GrowthKineticsEngine import GrowthKineticsEngine

    patient_data = Patient.Patient(indiv_name=args.indiv_id)
    mcmc_trace_cell_abundance, num_itertaions = load_mcmc_trace_abundances(args.abundance_mcmc_trace)
    with open(args.sif, 'r') as f:
        header = f.readline().strip('\n').split('\t')
        sample_time_points = {}
        for line in f:
            fields = dict(zip(header, line.strip('\n').split('\t')))
            try:
                sample_time_points[fields['sample_id']] = float(fields['timepoint'])
            except ValueError:
                sample_time_points = None
                break
    wbc_time_points = np.array(args.time) if args.time is not None else None
    gk_engine = GrowthKineticsEngine(patient_data, args.wbc)
    gk_engine.estimate_growth_rate(mcmc_trace_cell_abundance, wbc_time_points=wbc_time_points,
                                   sample_time_points=sample_time_points, n_iter=min(num_itertaions, args.n_iter))

    # Output and visualization
    import output.PhylogicOutput
    phylogicoutput = output.PhylogicOutput.PhylogicOutput()
    phylogicoutput.write_growth_rate_tsv(gk_engine.growth_rates, args.indiv_id)
    phylogicoutput.plot_growth_rates(gk_engine.growth_rates, args.indiv_id)


def load_mcmc_trace_abundances(in_file):
    iterations = set()
    cell_abundance_mcmc_trace = defaultdict(lambda: defaultdict(list))
    with open(in_file, 'r') as reader:
        for line in reader:
            values = line.strip('\n').split('\t')
            if line.startswith('Patient_ID'):
                header = {k: v for v, k in enumerate(values)}
            else:
                sample_id = values[header['Sample_ID']]
                cluster_id = int(values[header['Cluster_ID']])
                abundance = int(values[header['Abundance']])
                iterations.add(values[header['Iteration']])
                cell_abundance_mcmc_trace[sample_id][cluster_id].append(abundance)
    return cell_abundance_mcmc_trace, len(iterations)
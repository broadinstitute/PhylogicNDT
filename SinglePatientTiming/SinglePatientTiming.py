import logging
import numpy as np
import itertools
from data.Patient import Patient
import TimingEngine
from output.PhylogicOutput import PhylogicOutput


def run_tool(args):
    logging.debug('Arguments {}'.format(args))

    patient_data = Patient(artifact_blacklist=args.artifact_blacklist,
                           indiv_name=args.indiv_id, artifact_whitelist=args.artifact_whitelist,
                           driver_genes_file=args.driver_genes_file)

    if args.sif:  # if sif file is specified
        with open(args.sif, 'r') as sif_file:
            header = sif_file.readline().strip('\n').split('\t')
            for line in sif_file:
                row = dict(zip(header, line.strip('\n').split('\t')))
                sample_id = row['sample_id']
                maf_fn = row['maf_fn']
                seg_fn = row['seg_fn']
                purity = float(row['purity'])
                timepoint = float(row['timepoint'])
                patient_data.addSample(maf_fn, sample_id, input_type='post-clustering', seg_input_type='timing_format',
                                       timepoint_value=timepoint, grid_size=101, _additional_muts=None, seg_file=seg_fn,
                                       purity=purity)

    else:  # if sample names/files are specified directly on cmdline

        # sort order on timepoint or order of entry on cmdline if not present
        print(args.sample_data)
        for idx, sample_entry in enumerate(args.sample_data):
            ##for now, assume input order is of the type sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint
            smpl_spec = sample_entry.strip('\n').split(':')
            sample_id = smpl_spec[0]
            maf_fn = smpl_spec[1]
            seg_fn = smpl_spec[2]
            purity = float(smpl_spec[3])
            timepoint = float(smpl_spec[4])

            patient_data.addSample(maf_fn, sample_id, input_type='post-clustering', seg_input_type='timing_format',
                                   timepoint_value=timepoint, grid_size=args.grid_size,
                                   _additional_muts=None, seg_file=seg_fn, purity=purity)

    patient_data.preprocess_samples()
    if args.min_supporting_muts < 1:
        raise ValueError('Invalid value for min_supporting_muts')
    timing_engine = TimingEngine.TimingEngine(patient_data, min_supporting_muts=args.min_supporting_muts)
    timing_engine.time_events()
    phylogicoutput = PhylogicOutput()
    phylogicoutput.write_timing_tsv(timing_engine)
    with open(args.driver_genes_file) as f:
        driver_genes = [line.strip() for line in f]
    comps = compare_events(timing_engine, drivers=driver_genes)
    phylogicoutput.write_comp_table(args.indiv_id, comps)
    # phylogicoutput.generate_html_from_timing(args.indiv_id, timing_engine, comps, drivers=driver_genes)


def compare_events(timing_engine, drivers=()):
    """
    Compare event timings using pi distributions (convolutional difference compared to 0)
    """
    all_events = []
    if timing_engine.WGD:
        all_events.append(timing_engine.WGD)
    all_events.extend(itertools.chain(*timing_engine.all_cn_events.values()))
    all_events.extend(mut for mut in timing_engine.mutations.values() if
                      mut.prot_change and mut.gene in drivers and (mut.prot_change[0] != mut.prot_change[-1]
                      or not mut.prot_change[-1].isalpha()))
    comps = {}
    for eve1, eve2 in itertools.combinations(all_events, 2):
        if eve1.pi_dist is None or eve2.pi_dist is None:
            continue
        pi_diff = eve1.pi_dist - eve2.pi_dist
        p_before = 0.
        p_after = 0.
        diff_sign = 0.
        for d in pi_diff:
            if d > 0:
                if diff_sign < 0:
                    break
                p_before += d
                diff_sign = 1
            if d < 0:
                if diff_sign > 0:
                    break
                p_after -= d
                diff_sign = -1
        for d in reversed(pi_diff):
            if d > 0:
                if diff_sign < 0:
                    break
                p_after += d
                diff_sign = 1
            if d < 0:
                if diff_sign > 0:
                    break
                p_before -= d
                diff_sign = -1
        p_unknown = 1. - p_before - p_after
        # convoluted_dist = np.zeros(201)
        # for z in range(201):
        #     start1 = max(z - 100, 0)
        #     start2 = max(100 - z, 0)
        #     convoluted_dist[z] = np.inner(eve1.pi_dist[start1:z + 1], eve2.pi_dist[start2:201 - z])
        # convoluted_dist /= sum(convoluted_dist)
        # prob_event1_before_event2 = sum(convoluted_dist[:95])
        # prob_unknown = sum(convoluted_dist[95:106])
        # prob_event2_before_event1 = sum(convoluted_dist[106:])
        comps[(eve1.event_name, eve2.event_name)] = (p_before, p_after, p_unknown) # p_before refers to probability that event1 comes before event 2
    return comps

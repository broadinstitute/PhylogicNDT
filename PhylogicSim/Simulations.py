import warnings


def run_tool(args):
    from . import SimEngine

    if args.nsamp < 2:
        print("Minimum number of samples to simulate is 2, resetting to 2 samples.")
        args.nsamp = 2

    # TODO: Allow for only the clonal cluster to be simulated (args.min_nodes=1)
    if args.min_nodes < 2:
        warnings.warn('At least 2 clones required, will allow only clonal mutations in future versions', FutureWarning)
        args.min_nodes = 2

    min_number_muts = args.min_nodes * 3
    if args.nmuts < min_number_muts:
        warnings.warn('Number of mutations (%d) to simulate too small. Resetting to 3 mutations per cluster, %s' % (
            args.nmuts, min_number_muts))
        args.nmuts = min_number_muts

    SimEngine.run_simulations(args)

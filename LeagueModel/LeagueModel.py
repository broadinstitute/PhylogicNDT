from .LeagueModelData import Eve_Pair, Season, League

def run_league_model(args):
    import pandas as pd
    import cPickle as pkl
    import random
    import matplotlib.pyplot as plt
    import logging

    if args.comparison_fn is not None:
        full_comp_fn = args.comparison_fn
    elif args.comparison_fn is None and args.comps is not None:
        full_comp_fn = args.cohort + '.comparisons.tsv'
        output = open(full_comp_fn, 'w')
        output.write('\t'.join(['sample', 'event1', 'event2', 'p_event1_win', 'p_event2_win', 'unknown']) + '\n')
        for fn in args.comps:
            with open(fn, 'r') as input_fn:
                for i, row in enumerate(input_fn):
                    if i == 0: continue
                    output.write(row)
                output.write('\n')
        output.close()
    else:
        logging.error('Please Provide Input Data for League Model')
        return 0

    query_res_df1 = pd.read_csv(full_comp_fn, sep='\t')
    final_samples = None
    if args.final_sample_list is not None:
        final_samples = []
        with open(args.final_sample_list, 'r') as final_sample_list_fn:
            for i, row in enumerate(final_sample_list_fn):
                final_samples.append(row.strip("\n"))
    league_model_run = LeagueModelData.League(query_res_df1, cohort=args.cohort,
                                                       keep_only_samples_w_event=args.keep_samps_w_event,
                                                       remove_samps_w_event=args.remove_samps_w_event,
                                                       num_games_against_each_opponent=args.num_games_against_each_opponent,
                                                       final_samples=final_samples)
    output = open(args.cohort + '.all_events.tsv', 'w')
    output.write('indiv\tevent\n')
    for indiv in league_model_run.events_per_samp_full:
        for event in league_model_run.events_per_samp_full[indiv]:
            output.write(indiv + '\t' + event + '\n')
    output.close()

    all_samps = set(league_model_run.events_per_samp.keys())
    league_model_run.run_full_run(num_seasons=0, samples=all_samps, final_event_list=league_model_run.final_event_list)
    league_model_run.init_odds(league_model_run.final_event_list)

    ## output ##
    out_fn = args.cohort + '.matrix_of_comparisons.tsv'
    sorted_final_event_list = sorted(league_model_run.final_event_list)
    rev_list = sorted_final_event_list[::-1]
    with open(out_fn, 'w') as output:
        output.write('event_v_event\t' + '\t'.join(sorted_final_event_list) + '\n')
        for i, event1 in enumerate(rev_list):
            output.write(event1 + '\t')
            for j, event2 in enumerate(sorted_final_event_list):
                if event1 == event2:
                    output.write("na")
                else:
                    event_pair = tuple(sorted([event2, event1]))
                    if j <= len(sorted_final_event_list) - i - 1:
                        output.write(','.join(map(str, [league_model_run.event_pairs[event_pair].win_rates[event1],
                                                        league_model_run.event_pairs[event_pair].win_rates[event2],
                                                        league_model_run.event_pairs[event_pair].win_rates[
                                                            'unknown']])))
                    else:
                        output.write(','.join(map(str, [league_model_run.event_pairs[event_pair].win_rates[event2],
                                                        league_model_run.event_pairs[event_pair].win_rates[event1],
                                                        league_model_run.event_pairs[event_pair].win_rates[
                                                            'unknown']])))
                if j < len(sorted_final_event_list) - 1:
                    output.write('\t')
            output.write('\n')

    for j in range(args.n_perms):
        logging.info('running iter:' + str(j))

        rand_subset = set(random.sample(all_samps, int(len(all_samps) * args.percent_subset)))
        league_model_run.run_permutation(num_seasons=args.n_seasons, samples=rand_subset,
                                         final_event_list=league_model_run.final_event_list)
        league_model_run.update_odds()

    league_model_run.calc_log_odds_full_run()

    ## plotting ##
    league_model_run.plot_league_run(type='odds')
    fig = league_model_run.odds_plot
    plt.savefig(args.cohort + '.log_odds.pdf', transparent=True)
    plt.savefig(args.cohort + '.log_odds.png')
    league_model_run.plot_league_run(type='pos')
    fig = league_model_run.pos_plot
    plt.savefig(args.cohort + '.positions.pdf', transparent=True)
    plt.savefig(args.cohort + '.positions.png')

    ## output ##
    out_fn = args.cohort + '.prevalence.tsv'
    with open(out_fn, 'w') as output:
        output.write('cohort\tevent\tn_occur\tevent_split\ttype\tn_samp\n')
        for eve in league_model_run.final_event_list:
            if args.keep_samps_w_event is not None:
                output.write(args.cohort + '\t' + eve + '\t' + str(
                    league_model_run.full_event_prev[eve]) + '\t' + args.keep_samps_w_event +
                             '\twith\t' + str(league_model_run.full_n_samps) + '\n')
            elif args.remove_samps_w_event is not None:
                output.write(args.cohort + '\t' + eve + '\t' + str(
                    league_model_run.full_event_prev[eve]) + '\t' + args.remove_samps_w_event +
                             '\twithout\t' + str(league_model_run.full_n_samps) + '\n')
            else:
                output.write(
                    args.cohort + '\t' + eve + '\t' + str(league_model_run.full_event_prev[eve]) + '\tNone\tna\t' +
                    str(league_model_run.full_n_samps) + '\n')

    out_fn = args.cohort + '.full_prevalence.tsv'
    with open(out_fn, 'w') as output:
        output.write('cohort\tevent\tn_occur\tn_samp\n')
        for eve in league_model_run.full_event_prev:
            output.write(args.cohort + '\t' + eve + '\t' + str(league_model_run.full_event_prev[eve]) + '\t' + str(
                league_model_run.full_n_samps) + '\n')

    out_fn = args.cohort + '.log_odds.tsv'
    with open(out_fn, 'w') as output:
        output.write('cohort\tevent\tevent_split\ttype\tperm_run\tlog_odds_early\n')
        for eve in league_model_run.log_odds_full_run.keys():
            for j, odds in enumerate(league_model_run.log_odds_full_run[eve]):
                if args.keep_samps_w_event is not None:
                    output.write(args.cohort + '\t' + eve + '\t' + str(j) + '\t' + str(
                        league_model_run.log_odds_full_run[eve][j]) +
                                 '\t' + args.keep_samps_w_event + '\twith\t' + '\n')
                elif args.remove_samps_w_event is not None:
                    output.write(args.cohort + '\t' + eve + '\t' + str(j) + '\t' + str(
                        league_model_run.log_odds_full_run[eve][j]) +
                                 '\t' + args.remove_samps_w_event + '\twithout\t' + '\n')
                else:
                    output.write(args.cohort + '\t' + eve + '\t' + str(j) + '\t' + str(
                        league_model_run.log_odds_full_run[eve][j]) +
                                 '\tNone\tna\t' + '\n')

    pkl.dump(league_model_run, open(args.cohort + 'fig_pkl.pkl', 'wb'))


def run_league_model_ipython_notebook(args):
    import pandas as pd
    import cPickle as pkl
    import random
    import matplotlib.pyplot as plt
    import logging

    if args.comparison_fn is not None:
        full_comp_fn = args.comparison_fn
    elif args.comparison_fn is None and args.comps is not None:
        full_comp_fn = args.cohort + '.comparisons.tsv'
        output = open(full_comp_fn, 'w')
        output.write('\t'.join(['sample', 'event1', 'event2', 'p_event1_win', 'p_event2_win', 'unknown']) + '\n')
        for fn in args.comps:
            with open(fn, 'r') as input_fn:
                for i, row in enumerate(input_fn):
                    if i == 0: continue
                    output.write(row)
        output.close()
    else:
        logging.error('Please Provide Input Data for League Model')
        return 0

    query_res_df1 = pd.read_csv(full_comp_fn, sep='\t')
    final_samples = None
    if args.final_sample_list is not None:
        final_samples = []
        with open(args.final_sample_list, 'r') as final_sample_list_fn:
            for i, row in enumerate(final_sample_list_fn):
                final_samples.append(row.strip("\n"))
    league_model_run = LeagueModelData.League(query_res_df1, cohort=args.cohort,
                                                       keep_only_samples_w_event=args.keep_samps_w_event,
                                                       remove_samps_w_event=args.remove_samps_w_event,
                                                       num_games_against_each_opponent=args.num_games_against_each_opponent,
                                                       final_samples=final_samples)
    output = open(args.cohort + '.all_events.tsv', 'w')
    output.write('indiv\tevent\n')
    for indiv in league_model_run.events_per_samp_full:
        for event in league_model_run.events_per_samp_full[indiv]:
            output.write(indiv + '\t' + event + '\n')
    output.close()

    all_samps = set(league_model_run.events_per_samp.keys())
    league_model_run.run_full_run(num_seasons=0, samples=all_samps, final_event_list=league_model_run.final_event_list)
    league_model_run.init_odds(league_model_run.final_event_list)

    ## output ##
    out_fn = args.cohort + '.matrix_of_comparisons.tsv'
    sorted_final_event_list = sorted(league_model_run.final_event_list)
    rev_list = sorted_final_event_list[::-1]
    with open(out_fn, 'w') as output:
        output.write('event_v_event\t' + '\t'.join(sorted_final_event_list) + '\n')
        for i, event1 in enumerate(rev_list):
            output.write(event1 + '\t')
            for j, event2 in enumerate(sorted_final_event_list):
                if event1 == event2:
                    output.write("na")
                else:
                    event_pair = tuple(sorted([event2, event1]))
                    if j <= len(sorted_final_event_list) - i - 1:
                        output.write(','.join(map(str, [league_model_run.event_pairs[event_pair].win_rates[event1],
                                                        league_model_run.event_pairs[event_pair].win_rates[event2],
                                                        league_model_run.event_pairs[event_pair].win_rates[
                                                            'unknown']])))
                    else:
                        output.write(','.join(map(str, [league_model_run.event_pairs[event_pair].win_rates[event2],
                                                        league_model_run.event_pairs[event_pair].win_rates[event1],
                                                        league_model_run.event_pairs[event_pair].win_rates[
                                                            'unknown']])))
                if j < len(sorted_final_event_list) - 1:
                    output.write('\t')
            output.write('\n')

    for j in range(args.n_perms):
        logging.info('running iter:' + str(j))

        rand_subset = set(random.sample(all_samps, int(len(all_samps) * args.percent_subset)))
        league_model_run.run_permutation(num_seasons=args.n_seasons, samples=rand_subset,
                                         final_event_list=league_model_run.final_event_list)
        league_model_run.update_odds()

    league_model_run.calc_log_odds_full_run()

    ## plotting ##
    league_model_run.plot_league_run(type='odds')
    odds_plot = league_model_run.odds_plot
    plt.savefig(args.cohort + '.log_odds.pdf', transparent=True)
    plt.savefig(args.cohort + '.log_odds.png')
    league_model_run.plot_league_run(type='pos')
    pos_plot = league_model_run.pos_plot
    plt.savefig(args.cohort + '.positions.pdf', transparent=True)
    plt.savefig(args.cohort + '.positions.png')

    ## output ##
    out_fn = args.cohort + '.prevalence.tsv'
    with open(out_fn, 'w') as output:
        output.write('cohort\tevent\tn_occur\tevent_split\ttype\tn_samp\n')
        for eve in league_model_run.final_event_list:
            if args.keep_samps_w_event is not None:
                output.write(args.cohort + '\t' + eve + '\t' + str(
                    league_model_run.full_event_prev[eve]) + '\t' + args.keep_samps_w_event +
                             '\twith\t' + str(league_model_run.full_n_samps) + '\n')
            elif args.remove_samps_w_event is not None:
                output.write(args.cohort + '\t' + eve + '\t' + str(
                    league_model_run.full_event_prev[eve]) + '\t' + args.remove_samps_w_event +
                             '\twithout\t' + str(league_model_run.full_n_samps) + '\n')
            else:
                output.write(
                    args.cohort + '\t' + eve + '\t' + str(league_model_run.full_event_prev[eve]) + '\tNone\tna\t' +
                    str(league_model_run.full_n_samps) + '\n')

    out_fn = args.cohort + '.full_prevalence.tsv'
    with open(out_fn, 'w') as output:
        output.write('cohort\tevent\tn_occur\tn_samp\n')
        for eve in league_model_run.full_event_prev:
            output.write(args.cohort + '\t' + eve + '\t' + str(league_model_run.full_event_prev[eve]) + '\t' + str(
                league_model_run.full_n_samps) + '\n')

    out_fn = args.cohort + '.log_odds.tsv'
    with open(out_fn, 'w') as output:
        output.write('cohort\tevent\tevent_split\ttype\tperm_run\tlog_odds_early\n')
        for eve in league_model_run.log_odds_full_run.keys():
            for j, odds in enumerate(league_model_run.log_odds_full_run[eve]):
                if args.keep_samps_w_event is not None:
                    output.write(args.cohort + '\t' + eve + '\t' + str(j) + '\t' + str(
                        league_model_run.log_odds_full_run[eve][j]) +
                                 '\t' + args.keep_samps_w_event + '\twith\t' + '\n')
                elif args.remove_samps_w_event is not None:
                    output.write(args.cohort + '\t' + eve + '\t' + str(j) + '\t' + str(
                        league_model_run.log_odds_full_run[eve][j]) +
                                 '\t' + args.remove_samps_w_event + '\twithout\t' + '\n')
                else:
                    output.write(args.cohort + '\t' + eve + '\t' + str(j) + '\t' + str(
                        league_model_run.log_odds_full_run[eve][j]) +
                                 '\tNone\tna\t' + '\n')

    pkl.dump(league_model_run, open(args.cohort + 'fig_pkl.pkl', 'wb'))
    return odds_plot, pos_plot
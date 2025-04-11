## imports ##
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import numpy as np
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
import operator
from scipy.stats import gaussian_kde
import itertools
import copy
import logging

class Eve_Pair():

    def __init__(self, event1, event2, event1_type, event2_type):
        self.event1 = event1
        self.event_type1 = event1_type
        self.event2 = event2
        self.event_type2 = event2_type
        self.win_rates = {event1:0,event2:0,'unknown':0}
        self._hash = hash(":".join([self.event1, self.event2]))

    def calculate_rates(self):
        total_cooccurrences = sum(self.win_rates.values())
        self.num_cooccur = total_cooccurrences
        if total_cooccurrences > 0:
            self.mut1_win_rate = float(self.win_rates[self.event1]) / total_cooccurrences
            self.mut2_win_rate = float(self.win_rates[self.event2]) / total_cooccurrences
            self.draw_rate = float(self.win_rates['unknown']) / total_cooccurrences

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other

class Season():

    def __init__(self, league, season_number):
        self.league = league
        self.table = {}
        for event in self.league.final_event_list:
            self.table[event] = 0
        self._hash = hash(":".join(self.league.final_event_list) + ":" + str(season_number))

    def __hash__(self):
        return self._hash

    def update_table_from_multinomial_sampling(self, event_pair, multinomial_draw):
        # 2 points for a win, 1 point a draw, 0 points for a loss
        if multinomial_draw == [1, 0, 0]:
            self.table[event_pair[0]] += 2
        elif multinomial_draw == [0, 1, 0]:
            self.table[event_pair[1]] += 2
        else:
            self.table[event_pair[0]] += 1
            self.table[event_pair[1]] += 1

    def get_sorted_league(self):
        self.sorted_league = sorted(list(self.table.items()), key=operator.itemgetter(1), reverse=True)

    def get_event_positions(self):
        self.league_event_pos = {}
        prev_score = -1
        pos = 1
        pos_counter = 1
        for j,(eve,score) in enumerate(self.sorted_league):
            if score == prev_score:
                self.league_event_pos[eve] = pos
                pos_counter += 1
            else:
                self.league_event_pos[eve] = pos_counter
                prev_score = score
                pos_counter += 1
                pos = pos_counter

    def extra_time(self, event1, event2):
        return 0

    def pk_shootout(self, event1, event2):
        return 0

class League():

    ## params ##
    arm_CNVs = set([])
    for chrom in map(str, list(range(23))):
        arm_CNVs.add('loss_' + chrom + 'p')
        arm_CNVs.add('gain_' + chrom + 'p')
        arm_CNVs.add('loss_' + chrom + 'q')
        arm_CNVs.add('gain_' + chrom + 'q')
        arm_CNVs.add('loss_' + chrom)
        arm_CNVs.add('gain_' + chrom)
    arm_CNVs.add('WGD')

    def __init__(self,query_res_df,cohort=None,final_event_list=None,keep_only_samples_w_event=None,
                 remove_samps_w_event=None,keep_samps=None,remove_samps=None,num_games_against_each_opponent=2,final_samples=None):

        self.query_res_df = query_res_df
        self.seasons = []
        self.odds = {}
        self.event_pos = []
        self.cohort = cohort
        self.num_games_against_each_opponent = num_games_against_each_opponent
        logging.info('loading df')
        self.load_df()
        logging.info('subsetting to earliest mut')
        self.subset_to_earliest_point_mut()
        if final_samples is None:
            self.full_n_samps = len(self.events_per_samp)
        else:
            self.full_n_samps = len(final_samples)
        if keep_only_samples_w_event is not None:
            self.remove_samps_wout_event(keep_only_samples_w_event)
        if remove_samps_w_event is not None:
            self.remove_samps_with_event(remove_samps_w_event)
        if keep_samps is not None:
            self.keep_samps(keep_samps)
        if remove_samps is not None:
            self.remove_samps(remove_samps)
        logging.info('updating pairwise probs')
        self.update_pairwise_probs()
        logging.info('getting final event list')
        if final_event_list is not None:
            self.final_event_list = final_event_list
            self.calc_event_occur(final_event_list=final_event_list)
        else:
            self.calc_event_occur()
            self.final_event_list = self.get_final_event_list()
        self.form_pairs_for_league_model()
        self.update_pairs_for_league_model()
        self.run_league_model_iter(num_seasons=1000)
        self.n_perms = 0 # n_perms updated when updating odds
        self._hash = self.cohort

    def __hash__(self):
        return self._hash

    def add_season(self,season):
        self.seasons.append(season)

    def calc_event_occur(self,samples=None,final_event_list=None):

        self.event_prev = {}
        if samples is not None: sample_list = samples
        else: sample_list = list(self.events_per_samp.keys())

        if final_event_list is not None:
            for eve in final_event_list:
                if eve not in self.event_prev:
                    self.event_prev[eve] = 0

        for samp in sample_list:
            for eve in self.events_per_samp[samp]:
                if eve not in self.event_prev:
                    self.event_prev[eve] = 0
                self.event_prev[eve] += 1

        self.n_samps = len(sample_list)

    def remove_samps_wout_event(self,event):

        for samp in list(self.events_per_samp.keys()):
            if event not in self.events_per_samp[samp]:
                del self.events_per_samp[samp]
                del self.events_per_samp_full[samp]
                del self.num_comp[samp]
                del self.event_pairs_per_samp_full[samp]

    def remove_samps_with_event(self,event):

        for samp in list(self.events_per_samp.keys()):
            if event in self.events_per_samp[samp]:
                del self.events_per_samp[samp]
                del self.events_per_samp_full[samp]
                del self.num_comp[samp]
                del self.event_pairs_per_samp_full[samp]

    def keep_samps(self,samps):

        n_samps = 0
        output = open(self.cohort+'.final_samples.tsv','w')
        for samp in list(self.events_per_samp.keys()):
            if samp not in samps:
                del self.events_per_samp[samp]
                del self.events_per_samp_full[samp]
                del self.num_comp[samp]
                del self.event_pairs_per_samp_full[samp]
            else:
                output.write(samp+'\n')
                n_samps += 1
        output.close()
        self.full_n_samps = n_samps

    def remove_samps(self,samps):

        for samp in list(self.events_per_samp.keys()):
            if samp in samps:
                del self.events_per_samp[samp]
                del self.events_per_samp_full[samp]
                del self.num_comp[samp]
                del self.event_pairs_per_samp_full[samp]

    def get_samp_matrix_split(self,event1,event2):

        samps_w_both = [samp for samp in list(self.events_per_samp.keys()) if event1 in self.events_per_samp[samp]
                        and event2 in self.events_per_samp[samp]]
        samps_w_event1_only = [samp for samp in list(self.events_per_samp.keys()) if event1 in self.events_per_samp[samp]
                        and event2 not in self.events_per_samp[samp]]
        samps_w_event2_only = [samp for samp in list(self.events_per_samp.keys()) if event1 not in self.events_per_samp[samp]
                        and event2 in self.events_per_samp[samp]]
        samps_w_neither = [samp for samp in list(self.events_per_samp.keys()) if event1 not in self.events_per_samp[samp]
                        and event2 not in self.events_per_samp[samp]]
        return samps_w_both, samps_w_event1_only, samps_w_event2_only, samps_w_neither

    def load_df(self):

        self.events_per_samp = {}
        self.events_per_samp_full = {}
        self.event_pairs_per_samp_full = {}
        self.num_comp = {}
        self.mut_type = {}
        self.gene_names = {}

        for i, row in self.query_res_df.iterrows():

            samp = row['sample']
            event1 = row['event1']
            event2 = row['event2']
            p_event1_win = float(row['p_event1_win'])
            p_event2_win = float(row['p_event2_win'])
            p_unknown = float(row['unknown'])

            if samp not in self.events_per_samp:
                self.events_per_samp[samp] = set([])
                self.events_per_samp_full[samp] = set([])
                self.event_pairs_per_samp_full[samp] = {}
                self.num_comp[samp] = {}

            if event1.split("_")[0] in ['loss', 'gain', 'homdel']:event1_gene = "_".join(event1.split("_")[0:2])
            else:event1_gene = event1.split("_")[0].split(":")[0]
            if event2.split("_")[0] in ['loss', 'gain', 'homdel']:event2_gene = "_".join(event2.split("_")[0:2])
            else:event2_gene = event2.split("_")[0].split(":")[0]

            self.events_per_samp[samp].add(event1_gene)
            self.events_per_samp[samp].add(event2_gene)
            self.events_per_samp_full[samp].add(event1)
            self.events_per_samp_full[samp].add(event2)
            self.gene_names[event1] = event1_gene
            self.gene_names[event2] = event2_gene

            if event1 not in list(self.num_comp[samp].keys()):self.num_comp[samp][event1] = [0, 0]
            if event2 not in list(self.num_comp[samp].keys()):self.num_comp[samp][event2] = [0, 0]

            sorted_pair_full = tuple(sorted([event1, event2]))

            self.event_pairs_per_samp_full[samp][sorted_pair_full] = {(event1,event2):p_event1_win,(event2,event1):p_event2_win,'unknown':p_unknown}
            self.num_comp[samp][event1][0] += p_event1_win
            self.num_comp[samp][event2][1] += p_event2_win

            if event1_gene in self.arm_CNVs: self.mut_type[event1_gene] = 'arm_level'
            elif 'loss' in event1_gene or 'gain' in event1_gene or 'homdel' in event1_gene: self.mut_type[event1_gene] = 'focal_level'
            elif event1_gene == 'WGD': self.mut_type[event1_gene] = 'WGD'
            else: self.mut_type[event1_gene] = 'snv'

            if event2_gene in self.arm_CNVs: self.mut_type[event2_gene] = 'arm_level'
            elif 'loss' in event2_gene or 'gain' in event2_gene or 'homdel' in event2_gene: self.mut_type[event2_gene] = 'focal_level'
            elif event2_gene == 'WGD': self.mut_type[event2_gene] = 'WGD'
            else: self.mut_type[event2_gene] = 'snv'

    def get_samps_w_event(self,event):
        return [samp for samp in list(self.events_per_samp.keys()) if event in self.events_per_samp[samp]]

    # subsetting to earliest point mutations in cases of multi-hits
    def subset_to_earliest_point_mut(self):

        self.final_events_full = {}
        for samp in list(self.events_per_samp_full.keys()):

            self.final_events_full[samp] = []
            num_hits = {}
            point_muts = {}

            for mut in self.events_per_samp_full[samp]:
                if self.mut_type[self.gene_names[mut]] in ['arm_level','WGD','focal_level']:
                    self.final_events_full[samp].append(mut)
                else:
                    mut_gene = self.gene_names[mut]
                    if mut_gene not in num_hits:
                        num_hits[mut_gene] = 0
                        point_muts[mut_gene] = []
                    num_hits[mut_gene] += 1
                    point_muts[mut_gene].append(mut)

            for mut_gene in list(num_hits.keys()):
                if num_hits[mut_gene] == 1:
                    self.final_events_full[samp].append(point_muts[mut_gene][0])
                else:
                    earliest_mut = sorted([mut for mut in list(self.num_comp[samp].items()) if
                                           self.gene_names[mut[0]] == mut_gene],
                                          key = lambda x:(-x[1][1],x[1][0]))[0][0]
                    self.final_events_full[samp].append(earliest_mut)

    def update_pairwise_probs(self):

        self.event_pairs_per_samp = {}
        for samp in list(self.event_pairs_per_samp_full.keys()):
            self.event_pairs_per_samp[samp] = {}
            for eve_pair in list(self.event_pairs_per_samp_full[samp].keys()):

                eve1 = eve_pair[0]
                eve2 = eve_pair[1]
                eve1_gene = self.gene_names[eve1]
                eve2_gene = self.gene_names[eve2]
                gene_pair = tuple(sorted([eve1_gene,eve2_gene]))
                pair_probs = self.event_pairs_per_samp_full[samp][eve_pair]

                if eve1 in self.final_events_full[samp] and eve2 in self.final_events_full[samp]:

                    self.event_pairs_per_samp[samp][gene_pair] = {(eve1_gene,eve2_gene):pair_probs[(eve1,eve2)],
                                                                  (eve2_gene,eve1_gene):pair_probs[(eve2,eve1)],
                                                                  'unknown':pair_probs['unknown']}

    def get_final_event_list(self,max_mut=20,max_focal=5, max_homdel=5, num_gains_default=15,
                             num_losses_default=15,max_arm=30, min_prevalence=0.05):

        final_event_list = []

        num_arm = 0
        # first, add top occuring gains
        for i,(eve,n_occur) in enumerate(sorted([x for x in list(self.event_prev.items())
                                                 if self.mut_type[x[0]] in ['arm_level'] and
                                                    'gain' in x[0]], key = lambda y:y[1],reverse=True)):
            if i >= num_gains_default: break
            if n_occur/float(self.n_samps) < min_prevalence: break
            final_event_list.append(eve)
            num_arm += 1

        # then, add top occuring losses
        for i,(eve,n_occur) in enumerate(sorted([x for x in list(self.event_prev.items())
                                                 if self.mut_type[x[0]] in ['arm_level'] and
                                                    'loss' in x[0]], key = lambda y:y[1],reverse=True)):
            if i >= num_losses_default: break
            if n_occur/float(self.n_samps) < min_prevalence: break
            final_event_list.append(eve)
            num_arm += 1

        # then, add rest of arm events
        for i,(eve,n_occur) in enumerate(sorted([x for x in list(self.event_prev.items())
                                                 if self.mut_type[x[0]] in ['arm_level'] and
                                                    x[0] not in final_event_list], key = lambda y:y[1],reverse=True)):
            if i+num_arm >= max_arm: break
            if n_occur/float(self.n_samps) < min_prevalence: break
            final_event_list.append(eve)

        # then, add snvs/indels
        for i,(eve,n_occur) in enumerate(sorted([x for x in list(self.event_prev.items())
                                                 if self.mut_type[x[0]] in ['snv']],
                                                key = lambda y:y[1],reverse=True)):
            if i >= max_mut: break
            if n_occur/float(self.n_samps) < min_prevalence: break
            final_event_list.append(eve)

        # then, add focal events
        for i,(eve,n_occur) in enumerate(sorted([x for x in list(self.event_prev.items())
                                                 if self.mut_type[x[0]] in ['focal_level'] and
                                                    'homdel' not in x[0]], key = lambda y:y[1],reverse=True)):
            if i >= max_focal: break
            if n_occur/float(self.n_samps) < min_prevalence: break
            final_event_list.append(eve)

        # then, add homdels
        for i,(eve,n_occur) in enumerate(sorted([x for x in list(self.event_prev.items())
                                                 if self.mut_type[x[0]] in ['focal_level'] and
                                                    'homdel' in x[0]], key = lambda y:y[1],reverse=True)):
            if i >= max_homdel: break
            if n_occur/float(self.n_samps) < min_prevalence: break
            final_event_list.append(eve)

        # then, add WGD if present
        if 'WGD' in self.event_prev:
            final_event_list.append('WGD')

        final_event_list = list(set(final_event_list))
        return final_event_list

    def form_pairs_for_league_model(self,final_event_list=None):

        self.event_pairs = {}
        if final_event_list is not None:
            events = final_event_list
        else:
            events = self.final_event_list
        for eve1,eve2 in itertools.combinations(events, 2):
            sorted_pair = tuple(sorted([eve1,eve2]))
            self.event_pairs[sorted_pair] = Eve_Pair(sorted_pair[0],sorted_pair[1],self.mut_type[eve1],self.mut_type[eve2])

    def update_pairs_for_league_model(self,samples=None):

        if samples is not None: sample_list = samples
        else: sample_list = list(self.event_pairs_per_samp.keys())

        for samp in sample_list:
            for pair in list(self.event_pairs_per_samp[samp].keys()):
                eve1,eve2 = pair
                sorted_pair = tuple(sorted([eve1,eve2]))
                if sorted_pair not in self.event_pairs: continue
                self.event_pairs[sorted_pair].win_rates[eve1] += self.event_pairs_per_samp[samp][pair][(eve1,eve2)]
                self.event_pairs[sorted_pair].win_rates[eve2] += self.event_pairs_per_samp[samp][pair][(eve2,eve1)]
                self.event_pairs[sorted_pair].win_rates['unknown'] += self.event_pairs_per_samp[samp][pair]['unknown']

        for eve_pair in list(self.event_pairs.keys()):
            eve1,eve2 = eve_pair
            if sum(self.event_pairs[eve_pair].win_rates.values()) < 2:
                if self.event_prev[eve1] >= 4 and self.event_prev[eve2] >= 4:
                    self.event_pairs[eve_pair].win_rates['unknown'] += 1
                else:
                    self.event_pairs[eve_pair].win_rates[eve1] += 1
                    self.event_pairs[eve_pair].win_rates[eve2] += 1

        for eve_pair in list(self.event_pairs.keys()):
            self.event_pairs[eve_pair].calculate_rates()

    def run_league_model_iter(self,num_seasons):

        self.event_positions = {}
        for eve in self.final_event_list:
            self.event_positions[eve] = []

        for j,season in enumerate(range(num_seasons)):

            new_season = Season(self,season)
            for k in range(self.num_games_against_each_opponent):
                for event_pair in list(self.event_pairs.keys()):
                    #if event_pair[0] not in final_event_list or event_pair[1] not in final_event_list: continue
                    multinomial_draw = list(np.random.multinomial(1, [self.event_pairs[event_pair].mut1_win_rate,
                                                                      self.event_pairs[event_pair].mut2_win_rate,
                                                                      self.event_pairs[event_pair].draw_rate], size=1)[0])
                    new_season.update_table_from_multinomial_sampling(event_pair, multinomial_draw)

            new_season.get_sorted_league()
            new_season.get_event_positions()
            for eve in list(new_season.league_event_pos.keys()):
                self.event_positions[eve].append(new_season.league_event_pos[eve])
            self.add_season(season)

    def calc_odds(self):

        h1 = len(self.final_event_list) / 2.
        odds_dict = {}
        arr_final_pos = np.array([])

        for event in list(self.event_positions.keys()):
            hist = np.array(self.event_positions[event])
            arr_final_pos = np.concatenate((arr_final_pos, hist), axis=0)
            #odds_early = ( max(float(np.size(np.where(hist < q1))),1.) / (
            #max(float(np.size(np.where(hist >= q1))), 1)) ) / (max(float(np.size(np.where(hist > q4))),1.) / (
            #max(float(np.size(np.where(hist <= q4))), 1)) )

            odds_early = max(float(np.size(np.where(hist < h1))),1.) / max(float(np.size(np.where(hist >= h1))),1.)
            odds_late = max(float(np.size(np.where(hist >= h1))),1.) / max(float(np.size(np.where(hist < h1))),1.)

            #odds_early = max(float(np.size(np.where(hist <= q1))),1.) / max(float(np.size(np.where(hist >= q4))),1.)
            #odds_late = max(float(np.size(np.where(hist >= q4))),1.) / max(float(np.size(np.where(hist <= q1))),1.)
            #odds_late = 1./odds_early
            odds_dict[event] = {'odds_early':odds_early,'odds_late':odds_late}

        return odds_dict

    def init_odds(self,events):
        for eve in events:
            self.odds[eve] = {'odds_early':[],'odds_late':[]}

    def run_permutation(self,num_seasons,samples=None,final_event_list=None):

        self.num_seasons = num_seasons # <-- TODO: don't update every time, limits odds score in each iteration
        self.seasons = [] # <-- need to reset seasons
        self.event_pos = []
        self.calc_event_occur(samples,final_event_list)
        self.form_pairs_for_league_model(final_event_list)
        self.update_pairs_for_league_model(samples)
        self.run_league_model_iter(num_seasons=num_seasons)

    def update_odds(self):

        odds_dict = self.calc_odds()
        for event in list(odds_dict.keys()):
            self.odds[event]['odds_early'].append(odds_dict[event]['odds_early'])
            self.odds[event]['odds_late'].append(odds_dict[event]['odds_late'])
        self.n_perms += 1

    def run_full_run(self,num_seasons,samples=None,final_event_list=None):

        self.run_permutation(num_seasons=num_seasons,samples=samples,final_event_list=final_event_list)
        odds_dict = self.calc_odds()
        self.odds_full_run = {}
        for event in list(odds_dict.keys()):
            self.odds_full_run[event] = {'odds_early':odds_dict[event]['odds_early'],
                                         'odds_late':odds_dict[event]['odds_late']}
        self.full_event_prev = copy.deepcopy(self.event_prev)

    def calc_log_odds_full_run(self):

        self.log_odds_full_run = {}
        for eve in self.final_event_list:
            self.log_odds_full_run[eve] = np.log(np.array(self.odds[eve]['odds_early']))/np.log(10)

    ####################
    ##### plotting #####
    ####################

    ## helper function ##
    def autolabelh(self, rects, labels=None):
        # attach some text labels
        for i, rect in enumerate(rects):
            height = rect.get_width()
            label = str(int(height)) if labels is None else str(labels[i])
            plt.text(height + 1.5, rect.get_xy()[1] + 0.3, str(label), ha='left', va='center', fontsize=10)

    def plot_league_run(self,type='odds'):

        if type == 'odds':
            odds_plot = plt.figure(figsize=(8.0, 10.0))
        elif type == 'pos':
            pos_plot = plt.figure(figsize=(8.0, 10.0))
        else:
            return 0

        # fig4 is the odds plot (no clonal bars)
        sns.set(font_scale=1.0)
        sns.set_style('white')
        gs = gridspec.GridSpec(1, 10,wspace=0.05, hspace=0.05)
        ax0 = plt.subplot(gs[1:8])
        #plt.gcf().subplots_adjust(left=0.15)

        if type == 'odds':
            sorted_medians = [(y[0],y[1]) for y in sorted([(eve,np.median(self.log_odds_full_run[eve]),
                                                            min(self.log_odds_full_run[eve]),
                                                            max(self.log_odds_full_run[eve]))
                                                           for eve in list(self.log_odds_full_run.keys())],
                                                          key = lambda x:(x[1],x[2],x[3]),reverse=True)]
        elif type == 'pos':
            sorted_medians = [(y[0],y[1]) for y in sorted([(eve,np.median(self.event_positions[eve]),
                                                            min(self.event_positions[eve]),
                                                            max(self.event_positions[eve]))
                                                           for eve in self.final_event_list],
                                                          key = lambda x:(x[1],x[2],x[3]))]


        colors_v = []
        for med in sorted_medians:
            if 'loss' in med[0]: colors_v.append("#3498db")
            elif 'homdel' in med[0]: colors_v.append(sns.xkcd_rgb["royal blue"])
            elif med[0] == 'WGD': colors_v.append("black")
            elif 'gain' in med[0]: colors_v.append("#e74c3c")
            else: colors_v.append(sns.xkcd_rgb["faded green"])

        colors_l = ['black'] * len(colors_v)
        if type == 'odds':
            x_in = np.linspace(np.math.log(1./self.num_seasons,10), np.math.log(self.num_seasons,10), 10000 + 1)
        elif type == 'pos':
            x_in = np.linspace(1, len(self.final_event_list), 10000 + 1)

        for j, med in enumerate(sorted_medians):

            lw = 1
            if type == 'odds':
                to_plot = -np.log(np.array(self.odds[med[0]]['odds_early']))/np.log(10)
            elif type == 'pos':
                to_plot = self.event_positions[med[0]]

            if max(to_plot) == min(to_plot):
                to_plot = list(to_plot)
                to_plot.append(min(to_plot) - 0.1)
                to_plot.append(max(to_plot) + 0.1)
                to_plot = np.array(to_plot)
            density = gaussian_kde(to_plot)
            density.covariance_factor = lambda: 0.4
            density._compute_covariance()
            y = density(x_in)
            y_new = y / np.sum(y)

            left_sum = 0
            left_idx = 0
            for i in range(len(x_in)):
                left_sum += y_new[i]
                if left_sum >= 0.000005: break
                else: left_idx = i
            right_sum = 0
            right_idx = len(x_in)-1
            for i in range(len(x_in)-1,-1,-1):
                right_sum += y_new[i]
                if right_sum >= 0.000005: break
                else: right_idx = i

            y_new = y_new / max(y_new) * 0.2
            plt.fill_between(x_in[left_idx:right_idx],y_new[left_idx:right_idx] + len(sorted_medians) - j - 1,
                             -y_new[left_idx:right_idx] + len(sorted_medians) - j - 1, color='gray',
                             alpha=0.8, lw=lw)

        sorted_medians.reverse()
        colors_v.reverse()

        ax0.set_yticks([i for i in range(0, len(sorted_medians))])
        ax0.set_yticklabels([str(" ".join(med[0].split("_"))) for med in sorted_medians],
                            fontsize=15. * 35 / max(30, len(self.final_event_list)))
        for ytick, color in zip(ax0.get_yticklabels(), colors_l): ytick.set_color(color)
        plt.title(self.cohort + ', N(samp) = ' + str(self.full_n_samps)+', N(events) = ' +
                  str(len(self.final_event_list)), fontsize=16)
        if type == 'odds':
            plt.xlabel('relative log odds timing', fontsize=16)
            plt.xlim(np.math.log(1./self.num_seasons,10)-0.1,np.math.log(self.num_seasons,10)+0.1)
        elif type == 'pos':
            plt.xlabel('event position', fontsize=16)
            plt.xlim(0, len(self.final_event_list)+1)
        plt.ylabel('event', fontsize=16)
        plt.ylim(-0.5, len(sorted_medians) - 0.5)
        ax2 = plt.subplot(gs[8:])
        plt.title('prevalence', fontsize=10)
        sns.set_style('white')
        bars = ax2.barh(np.array(list(range(len(sorted_medians)))),
                        [self.full_event_prev[event[0]] for event in sorted_medians], color=colors_v, align='center')
        labels = [str(int(round(
            [self.full_event_prev[event[0]] for event in
             sorted_medians][i] / float(self.full_n_samps), 2) * 100)) + "%" for i in range(len(sorted_medians))]

        self.autolabelh(bars, labels)
        plt.axis("off")
        plt.ylim(-0.5, len(sorted_medians) - 0.5)

        if type == 'odds':
            self.odds_plot = odds_plot
        elif type == 'pos':
            self.pos_plot = pos_plot

    ####################
    ###### outputs #####
    ####################
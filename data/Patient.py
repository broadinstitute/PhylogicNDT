##########################################################
## Patient - central class to handle and store sample CCF data
##########################################################
from Sample import TumorSample
from SomaticEvents import SomMutation, CopyNumberEvent
import os
import logging
import itertools
import functools
import numpy as np
from intervaltree import Interval, IntervalTree

#Logsumexp options
import pkgutil
if pkgutil.find_loader('sselogsumexp') is not None:
    from sselogsumexp import logsumexp
else:
    from scipy.misc import logsumexp # LogAdd



class Patient:
    """CLASS info

     FUNCTIONS:
         public:

           add_sample() - add new samples from individual to self.sample_list by making a new TumorSample instance

        private:

           _auto_file_type() - guess input file type for ccf data if not manually provided

     PROPERTIES:
         regular variables:
            self.sample_list

         @property methods: none
    """

    def __init__(self, indiv_name='Indiv1',
                 sample_map={},
                 ccf_grid_size = 101,
                 driver_genes_file=os.path.join(os.path.dirname(__file__), 'supplement_data/Driver_genes_v1.0.txt'),
                 impute_missing=False,
                 artifact_blacklist=os.path.join(os.path.dirname(__file__), 'supplement_data/Blacklist_SNVs.txt'),
                 artifact_whitelist='', use_indels=False,
                 min_coverage=8, delete_auto_bl=False, PoN_file=False):


        #DECLARATIONS
        #@properties


        self.indiv_name = indiv_name
        self.sample_list = []
        """ :type : list [TumorSample]"""

        self.samples_synchronized = False

        self.driver_genes = self._parse_driver_g_file(driver_genes_file)
        # hash table to store known driver genes #TODO add enum


        self.ccf_grid_size = ccf_grid_size

        # @annatotion
        self.PatientLevel_MutBlacklist = artifact_blacklist
        self.PatientLevel_MutWhitelist = artifact_whitelist

        #Patient configuration settings

        self.impute_missing = impute_missing  # flag if to impute missing variants as ccf 0
        #self.delete_auto_bl = delete_auto_bl
        self.min_coverage = min_coverage  # min cov is specified both here and passed to tumor sample.
        self.use_indels = use_indels
        self.PoN_file = PoN_file

        self._validate_sample_names()

        #later filled data objects

        self.ND_mutations=[]
        #@methods

        #storing of results
        #Clustering
        self.ClusteringResults=None

        self.MutClusters=None
        self.TruncalMutEvents=None
        self.MCMC_trace=None
        self.k_trace=None
        self.alpha_trace=None

        self.unclustered_muts = []

        # self.concordant_cn_events = []
        self.concordant_cn_tree = {chrom: IntervalTree() for chrom in list(map(str, range(1, 23)))+['X', 'Y']}

        #BuildTree
        self.TopTree=None
        self.TreeEnsemble=[]


    def initPatient(self):  # input_type abs; txt; sqlite3 .db # auto tab if .txt, .tsv or .tab ; abs if .Rdata; sqlite if .db
        raise NotImplementedError

    def addSample(self, filen, sample_name, input_type='auto', grid_size=101,
               indiv="Not_Set", seg_file=None,
               _additional_muts=None,
               purity=None,timepoint_value=None):  # input_type abs; txt; sqlite3 .db # auto tab if .txt, .tsv or .tab ; abs if .Rdata; sqlite if .db

        if _additional_muts == []:
            _additional_muts = None
        elif type(_additional_muts) is list:
            _additional_muts = _additional_muts[
                0]

        # make new sample and add to exiting list of samples
        logging.info("Adding Mutations from Sample: %s", sample_name)
        new_sample = TumorSample(filen, input_type, sample_name=sample_name, artifact_blacklist=self.PatientLevel_MutBlacklist,
                                 artifact_whitelist=self.PatientLevel_MutWhitelist,
                                 ccf_grid_size=grid_size, PoN=self.PoN_file, indiv=self.indiv_name,
                                 use_indels=self.use_indels, min_coverage=self.min_coverage,
                                 _additional_muts=_additional_muts, seg_file=seg_file,
                                 purity=purity,timepoint_value=timepoint_value)

        self.sample_list.append(new_sample)
        logging.info('Added sample ' + new_sample.sample_name)

        self.samples_synchronized = False  # turn of concordance flag when new sample is added

    #main clean and init function
    def homogenize_events_across_samples(self):
        if len(self.sample_list)==1:
            UserWarning("Found one sample! Will run 1D clustering. If intentended to run ND, please fix!")



    # \=======getter functions=============
    def get_sample_byname(self, sample_name):
            """
            :param str sample_name: sample name to search for
            :return: TumorSample
            """
            for smpl in self.sample_list:
                if smpl.sample_name == sample_name: return smpl
            return None

    def _validate_sample_names(self, disallowed_strings=['Inner']):

        sample_names = self.sample_names

        if len(sample_names) != len(set(sample_names)):  # check that no samples have the same name
            ValueError('Several samples appear to have identical names! This is not allowed! Please fix!')

        for dis_str in disallowed_strings:  # check that no samples have the same name
            for smpl_name in sample_names:
                if dis_str in smpl_name:
                    ValueError('Several samples appear to have identical names! This is not allowed! Please fix!')

        return True

    @property
    def sample_names(self):
            return [x.sample_name for x in self.sample_list]

    @staticmethod
    def _parse_driver_g_file(filen): #read driver file as one gene per line
            if not filen: return set() #empty list
            with open(filen) as drv_file:
                drv=[ x.strip() for x in drv_file.read().strip().split('\n')]
            return set(drv)

    # \==========central preprocessing of samples===========================
    def preprocess_samples(self):  #make sure there is info on all mutations and everything is concordant on all samples

        if len(self.sample_list) < 2:
            logging.warning("Only one sample in sample set! Cannot check concordance for multi-D clustering!")
            logging.warning ("Cleaning this single sample!")  #do logging

            ##TODO: port the 1D-preprocess
            for sample in self.sample_list:  # because of imputing: has to be done at the end
                sample.concordant_variants = sample.mutations
                sample.concordant_variants.sort(
                    key=lambda x: (str(x).split('_')[0] in self.driver_genes, str(x),
                                   x.var_str))  # sort concordant variants by presence in driver then var_str/gene name
                sample.private_mutations = [mut for mut in sample.concordant_variants if mut.alt_cnt > 0]

        # sys.exit(1)

        self._validate_sample_names()
        try:
            blacklist = set(set.union(*[set(x.known_blacklisted_mut) for x in self.sample_list]))
        except TypeError:
            logging.info("Blacklist not specified.")
            blacklist = set()
            whitelist = None

        try:
            whitelist = set(set.union(*[set(x._whitelist) for x in self.sample_list]))
        except TypeError:
            logging.info("Whitelist not specified.")
            whitelist = None

        # use var stings from sample1 to order all other samples

        var_str_init = self.sample_list[0].mut_varstr
        count_needed = len(self.sample_list)
        full_var_list = list(itertools.chain.from_iterable([x.mut_varstr for x in self.sample_list]))
        # print len(list(full_var_list))

        vars_present_in_all = set()
        for mut in var_str_init:
            if full_var_list.count(mut) == count_needed and mut not in blacklist:
                vars_present_in_all.add(mut)

        joint_temporarily_removed = set()
        for sample in self.sample_list:
            joint_temporarily_removed.update(sample.temporarily_removed)

        joint_temporarily_removed = set([mut.var_str for mut in joint_temporarily_removed])
        # iterate through all samples and make concordance sets, then sort them

        first_cnv = None  ## Only allow one fixed CNV in concordant variants.

        for sample in self.sample_list:
            if not self.impute_missing: sample.concordant_variants = []  # reset any previous joint results ; if imputing leave as is
            sample.concordant_with_samples = []

            for mut in sample.mutations + sample.low_coverage_mutations.values():

                # TODO: Add this as a flag
                # if mut.mut_category is not None and ("rna" in mut.mut_category.lower() or "utr" in mut.mut_category.lower()):
                #    logging.warning("REMOVING RNA MUTATION:" + str(mut))
                #    sample.artifacts_in_blacklist.append(mut)
                #    blacklist.add(mut.var_str)
                #

                if mut.var_str in blacklist:  # If blacklisted in another sample, remove mutation.
                    logging.info("Found blacklisted mutation still in sample, removing:" + str(mut))
                    if mut.var_str not in sample.known_blacklisted_mut:
                        sample.known_blacklisted_mut.add(mut.var_str)
                    continue

                elif mut.var_str in vars_present_in_all:

                    if mut.var_str not in joint_temporarily_removed:
                        if mut.type == "CNV":
                            if first_cnv is None:
                                first_cnv = mut.var_str
                        if mut.type != "CNV" or first_cnv == mut.var_str:
                            sample.concordant_variants.append(mut)

                    else:
                        continue  # nothing to do for indels in all samples

                elif mut.var_str not in vars_present_in_all:
                    if mut.var_str in joint_temporarily_removed:
                        sample.artifacts_in_blacklist.append(mut)
                        logging.warn("Forcecalling Failure? Muatation " + str(
                            mut) + " not in " + sample.sample_name + " but otherwise graylisted already.")
                        continue

                    elif mut.var_str not in joint_temporarily_removed and self.impute_missing:
                        for mis_sample in [x for x in self.sample_list if x != sample]:
                            if mut not in mis_sample.concordant_variants and mut not in mis_sample.mutations and mut.var_str not in joint_temporarily_removed and ":".join(
                                    map(str,
                                        [mut.chrN, mut.pos, mut.ref, mut.alt])) not in mis_sample.known_blacklisted_mut:
                                logging.info('Imputing missing mutation as 0 CCF, sample %s, var: %s ; %s',
                                             sample.sample_name, mut, mut.var_str)
                                mis_sample.concordant_variants.append(
                                    SomMutation.from_som_mutation_zero(mut, from_sample=mis_sample))
                                mis_sample._mut_varstring_hastable[mut.var_str] = mis_sample.concordant_variants[-1]
                        sample.concordant_variants.append(mut)

                    else:
                        logging.error(
                            'Mutation missing in some datasets, sample %s, var: %s . This mutation will be skipped. Use --impute, or forcecall the mutations.',
                            sample.sample_name, mut)

        for sample in self.sample_list:  # because of imputing: has to be done at the end
            sample.concordant_variants.sort(
                key=lambda x: (str(x).split('_')[0] in self.driver_genes, str(x),
                               x.var_str))  # sort concordant variants by presence in driver then var_str/gene name
            sample.concordant_with_samples = self.sample_list  # annotate each sample with what samples were used for concordance
        # print [(x,x.var_str) for x in sample.concordant_variants]
        # print sample.concordant_with_samples
        self.samples_synchronized = True  # turn on concordance flag when new sample mut are joined

    def _make_ND_histogram(self):
        if not self.samples_synchronized:
            logging.error("Could not make ND Histogram, please make sure to call preprocess_samples()")
            return False

        combined_ccf = []
        for mut in self.sample_list[0].concordant_variants:
            mut_nd_ccf = [mut.ccf_1d]
            for sample in self.sample_list[1:]:
                mut_nd_ccf.append(sample.get_mut_by_varstr(mut.var_str).ccf_1d)
            combined_ccf.append(mut_nd_ccf)

        return NDHistogram(combined_ccf, [x.var_str for x in self.sample_list[0].concordant_variants])

    def cluster_temp_removed(self):
        clust_CCF_results = self.ClusteringResults.clust_CCF_dens
        for mut in self.sample_list[0].low_coverage_mutations.values():
            mut_coincidence = np.ones(len(clust_CCF_results))
            for i, sample in enumerate(self.sample_list):
                try:
                    mut = sample.get_mut_by_varstr(mut.var_str)
                except KeyError:
                    logging.warning(mut.var_str + ' not called across all samples')
                    mut_coincidence.fill(np.nan)
                    break
                for ii, cluster_ccfs in enumerate(clust_CCF_results):
                    cluster_ccf = cluster_ccfs[i]
                    if abs(np.argmax(mut.ccf_1d) - np.argmax(cluster_ccf)) > 50:
                        dot = 0.
                    else:
                        dot = max(sum(np.array(mut.ccf_1d) * cluster_ccf), .0001)
                    mut_coincidence[ii] *= dot
            if np.any(mut_coincidence > 0.):
                cluster_assignment = np.argmax(mut_coincidence) + 1
                for i, sample in enumerate(self.sample_list):
                    mut = sample.get_mut_by_varstr(mut.var_str)
                    mut.cluster_assignment = cluster_assignment
                    mut.clust_ccf = clust_CCF_results[cluster_assignment - 1][i]
            elif not np.any(np.isnan(mut_coincidence)):
                self.unclustered_muts.append(mut)

    def intersect_cn_trees(self):
        def get_bands(chrom, start, end, cytoband=os.path.dirname(__file__) + '/supplement_data/cytoBand.txt'):
            bands = []
            on_c = False
            with open(cytoband, 'r') as f:
                for line in f:
                    row = line.strip('\n').split('\t')
                    if row[0].strip('chr') != str(chrom):
                        if on_c:
                            return bands
                        continue
                    if int(row[1]) <= end and int(row[2]) >= start:
                        bands.append(Cytoband(chrom, row[3]))
                        on_c = True
                    if int(row[1]) > end:
                        return bands

        def merge_cn_events(event_segs, neighbors, R=frozenset(), X=frozenset()):
            is_max = True
            for s in itertools.chain(event_segs, X):
                if isadjacent(s, R):
                    is_max = False
                    break
            if is_max:
                bands = set.union(*(set(b[0]) for b in R))
                cns = next(iter(R))[1]
                ccf_hat = np.zeros(len(self.sample_list))
                ccf_high = np.zeros(len(self.sample_list))
                ccf_low = np.zeros(len(self.sample_list))
                for seg in R:
                    ccf_hat += np.array(seg[2])
                    ccf_high += np.array(seg[3])
                    ccf_low += np.array(seg[4])
                yield (bands, cns, ccf_hat / len(R), ccf_high / len(R), ccf_low / len(R))
            else:
                for s in min((event_segs - neighbors[p] for p in event_segs.union(X)), key=len):
                    if isadjacent(s, R):
                        for region in merge_cn_events(event_segs.intersection(neighbors[s]), neighbors, R=R.union({s}),
                                                      X=X.intersection(neighbors[s])):
                            yield region
                        event_segs = event_segs.difference({s})
                        X = X.union({s})

        def isadjacent(s, R):
            if not R:
                return True
            Rchain = list(itertools.chain(*(b[0] for b in R)))
            minR = min(Rchain)
            maxR = max(Rchain)
            mins = min(s[0])
            maxs = max(s[0])
            if mins >= maxR:
                return mins - maxR <= 1 and mins.band[0] == maxR.band[0]
            elif maxs <= minR:
                return minR - maxs <= 1 and maxs.band[0] == minR.band[0]
            else:
                return False

        c_trees = {}
        n_samples = len(self.sample_list)
        for chrom in list(map(str, range(1, 23)))+['X', 'Y']:
            tree = IntervalTree()
            for sample in self.sample_list:
                if sample.CnProfile:
                    tree.update(sample.CnProfile[chrom])
            tree.split_overlaps()
            tree.merge_equals(data_initializer=[], data_reducer=lambda a, c: a + [c])
            c_tree = IntervalTree(filter(lambda s: len(s.data) == n_samples, tree))
            c_trees[chrom] = c_tree
            event_segs = set()
            for seg in c_tree:
                start = seg.begin
                end = seg.end
                bands = get_bands(chrom, start, end)
                cns_a1 = []
                cns_a2 = []
                ccf_hat_a1 = []
                ccf_hat_a2 = []
                ccf_high_a1 = []
                ccf_high_a2 = []
                ccf_low_a1 = []
                ccf_low_a2 = []
                for i, sample in enumerate(self.sample_list):
                    cns_a1.append(seg.data[i][1]['cn_a1'])
                    cns_a2.append(seg.data[i][1]['cn_a2'])
                    ccf_hat_a1.append(seg.data[i][1]['ccf_hat_a1'] if seg.data[i][1]['cn_a1'] != 1 else 0.)
                    ccf_hat_a2.append(seg.data[i][1]['ccf_hat_a2'] if seg.data[i][1]['cn_a2'] != 1 else 0.)
                    ccf_high_a1.append(seg.data[i][1]['ccf_high_a1'] if seg.data[i][1]['cn_a1'] != 1 else 0.)
                    ccf_high_a2.append(seg.data[i][1]['ccf_high_a2'] if seg.data[i][1]['cn_a2'] != 1 else 0.)
                    ccf_low_a1.append(seg.data[i][1]['ccf_low_a1'] if seg.data[i][1]['cn_a1'] != 1 else 0.)
                    ccf_low_a2.append(seg.data[i][1]['ccf_low_a2'] if seg.data[i][1]['cn_a2'] != 1 else 0.)
                cns_a1 = np.array(cns_a1)
                cns_a2 = np.array(cns_a2)
                if np.all(cns_a1 == 1):
                    pass
                elif np.all(cns_a1 >= 1) or np.all(cns_a1 <= 1):
                    event_segs.add((tuple(bands), tuple(cns_a1), tuple(ccf_hat_a1), tuple(ccf_high_a1), tuple(ccf_low_a1), 'a1'))
                else:
                    logging.warning('Seg with inconsistent event: {}:{}:{}'.format(chrom, seg.begin, seg.end))
                if np.all(cns_a2 == 1):
                    pass
                elif np.all(cns_a2 >= 1) or np.all(cns_a2 <= 1):
                    event_segs.add((tuple(bands), tuple(cns_a2), tuple(ccf_hat_a2), tuple(ccf_high_a2), tuple(ccf_low_a2), 'a2'))
                else:
                    logging.warning('Seg with inconsistent event: {}:{}:{}'.format(chrom, seg.begin, seg.end))
            neighbors = {s: set() for s in event_segs}
            for seg1, seg2 in itertools.combinations(event_segs, 2):
                s1_hat = np.array(seg1[2])
                s2_hat = np.array(seg2[2])
                if seg1[1] == seg2[1] and np.all(s1_hat >= np.array(seg2[4])) and np.all(s1_hat <= np.array(seg2[3]))\
                and np.all(s2_hat >= np.array(seg1[4])) and np.all(s2_hat <= np.array(seg1[3])):
                    neighbors[seg1].add(seg2)
                    neighbors[seg2].add(seg1)

            event_cache = []
            if event_segs:
                for bands, cns, ccf_hat, ccf_high, ccf_low in merge_cn_events(event_segs, neighbors):
                    mut_category = 'gain' if sum(cns) > len(self.sample_list) else 'loss'
                    a1 = (mut_category, bands) not in event_cache
                    if a1:
                        event_cache.append((mut_category, bands))
                    self._add_cn_event_to_samples(chrom, min(bands), max(bands), cns, mut_category, ccf_hat, ccf_high,
                        ccf_low, a1, dupe=not a1)
        self.concordant_cn_tree = c_trees

    def _add_cn_event_to_samples(self, chrom, start, end, cns, mut_category, ccf_hat, ccf_high, ccf_low, a1, dupe=False):
        for i, sample in enumerate(self.sample_list):
            local_cn = cns[i]
            ccf_hat_i = ccf_hat[i] if local_cn != 1. else 0.
            ccf_high_i = ccf_high[i] if local_cn != 1. else 0.
            ccf_low_i = ccf_low[i] if local_cn != 1. else 0.
            cn = CopyNumberEvent(chrom, start, end, ccf_hat=ccf_hat_i, ccf_high=ccf_high_i, ccf_low=ccf_low_i,
                                 local_cn=local_cn, from_sample=sample, a1=a1, mut_category=mut_category, dupe=dupe)
            sample.low_coverage_mutations.update({cn.var_str: cn})
            sample.add_muts_to_hashtable(cn)


#################################################################################
##NDHistogram() - helper class to store combined histograms of mutations to pass to DP
#################################################################################
class NDHistogram:
    """CLASS info

    FUNCTIONS:

    PROPERTIES:
    """

    def __init__(self, hist_array, labels, phasing=None, ignore_nan=False):
        conv = 1e-40

        ### We need to noramlize the array to 1. This is hard.
        ### This step does the following:
        ##### convert to log space after adding a small convolution paramter so we don't get INF and NAN
        ##### for each mutation
        ##### normalize the row to 1 in logspace.

        hist = np.asarray(hist_array, dtype=np.float32) + conv
        if (~(hist > 0)).any():
            logging.error("Negative histogram bin or NAN mutation!")
            if ignore_nan:
                logging.warning("Caught ignore nan flag, not exiting, setting nan values to zero")
                hist[np.logical_not(hist > 0)] = conv
            else:
                sys.exit(1)

        n_samples = np.shape(hist)[1]
        for sample in range(n_samples):
            hist[:,:,0]=conv ##set zero bins
        self._hist_array = np.apply_over_axes(lambda x, y: np.apply_along_axis(lambda z: z - logsumexp(z), y, x),
                                              np.log(hist), 2)
        ####



        self._labels = labels
        self._label_ids = dict([[y, x] for x, y in enumerate(labels)])
        self._phasing = {} if phasing is None else phasing
        self.n_samples = np.shape(hist_array)[1]
        self.n_bins = np.shape(hist_array)[-1]

    def __getitem__(self, key):
        return self._hist_array[self._label_ids[key]]

    def phase(self, m1, m2):
        return self._phasing[frozenset({m1, m2})]

    def iteritems(self):
        for idx, mut in enumerate(self._hist_array):
            yield self._labels[idx], mut

    @property
    def mutations(self):
        return {}


class Cytoband:

    def __init__(self, chrom, band):
        self.chrom = chrom
        self.band = band

    def __sub__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot subtract Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            self_num = -1
            other_num = -1
            with open(os.path.dirname(__file__)+'/supplement_data/cytoBand.txt', 'r') as f:
                for i, line in enumerate(f):
                    row = line.strip('\n').split('\t')
                    if row[0] == 'chr' + str(self.chrom) and row[3] == self.band:
                        self_num = i
                    if row[0] == 'chr' + str(other.chrom) and row[3] == other.band:
                        other_num = i
                    if self_num >= 0 and other_num >= 0:
                        return self_num - other_num
        raise ValueError('Cannot subtract cytobands on different chromosomes')

    def __lt__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            return self - other < 0
        raise ValueError('Cannot compare cytobands on different chromosomes')

    def __gt__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            return self - other > 0
        raise ValueError('Cannot compare cytobands on different chromosomes')

    def __le__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            return self - other <= 0
        raise ValueError('Cannot compare cytobands on different chromosomes')

    def __ge__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        if self.chrom == other.chrom:
            return self - other >= 0
        raise ValueError('Cannot compare cytobands on different chromosomes')

    def __eq__(self, other):
        if not isinstance(other, Cytoband):
            raise TypeError('Cannot compare Cytoband with ' + str(type(other)))
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.chrom, self.band))

    def __repr__(self):
        return str(self.chrom) + self.band



##############################################################

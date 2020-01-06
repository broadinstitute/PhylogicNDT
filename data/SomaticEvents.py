import os
import sys
import Enums
import logging
from intervaltree import IntervalTree

import numpy as np
import scipy.stats


##########################################################
# Patient's SomaticEvent - class to store "virtual" mutation on Patient level
##########################################################
class SomMutation:
    """CLASS info

        FUNCTIONS:

            public:

            private:

        PROPERTIES:

            regular variables:

            @property methods: none
    """

    def __init__(self, chrN, pos, ref, alt, ccf_1d, ref_cnt=None, alt_cnt=None, gene=None, prot_change=None,
                 mut_category=None, det_power=None, from_sample=None, type_=None,
                 clonality=None,
                 on_arm_event=None,
                 seg_tree=None,
                 clust_ccf=None,
                 cluster_assignment=None,
                 multiplicity=np.nan,
                 local_cn_a1=np.nan,
                 local_cn_a2=np.nan,
                 type_local_cn_event=np.nan,
                 clonal_cut_off=0.86):

        # try:
        # Link to sample object.
        self.from_sample = from_sample

        # Basic properties of mutation
        self.chrN = chrN
        self.pos = int(float(pos))
        self.ref = ref
        self.alt = alt
        self.ref_cnt = int(
            float(ref_cnt)) if ref_cnt is not None else None  # TODO: add support/handling for text CCF files
        self.alt_cnt = int(float(alt_cnt)) if alt_cnt is not None else None
        self.gene = gene if gene is not None and 'unknown' not in gene.lower() else None

        self.blacklist_status = False  # is blacklisted mutation
        self.graylist_status = False  # is blacklisted mutation

        # raw ccf before clustering
        self.ccf_1d = ccf_1d

        self.status = Enums.MutStatus.OK  # later if blacklisted or graylisted

        self.clean_local_cn(local_cn_a1, local_cn_a2)

        if type_ is None:
            # guess type for backwards compatibility, but prefer that type be passed explicitly.
            if "-" in self.ref:
                self.type = Enums.MutType.INS
            elif "-" in self.alt:
                self.type = Enums.MutType.DEL
            else:
                self.type = Enums.MutType.SNV
        else:
            self.type = type_

        self._var_str = ":".join(map(str, [self.chrN, self.pos, self.ref,
                                           self.alt]))  # variant string, set as private, gotten with self.var_str

        # try:
        #    self.ccf_1d = tuple(map(float, ccf_1d))  # make sure all ccf values are floats
        #    # TODO: CCF broadening - this needs to be implemented. Currently does nothing.
        #    if max(self.ccf_1d) > 0.25:  # hueristic - if 25% of the weight is in one bin, broaden it.
        #        self.ccf_1d = self._broaden_ccf(self.ccf_1d)
        # except:
        #    logging.debug("Cannot parse ccf_1d in SomMutation instantiation")
        #    self.ccf_1d = None

        # self.blacklist_status = False
        # self.graylist_status = False

        """
        TODO:auto set blacklist/graylist status.
        if self.from_sample is not None:
            if self.var_str in self.from_sample.known_blacklisted_mut:
                self.blacklist_status=True
        if self.from_sample
        """

        self.prot_change = prot_change if prot_change is not None and 'unknown' not in prot_change.lower() else None  # Unknown values are set as __UNKNOWN__ in ocotator, but Unknown is also something that should be skipped for gene names.

        # several values/placeholders for output options
        self.mut_category = mut_category
        self.color = None

        # move to output
        # This check is to avoid crashes when det_power is "NA" or a non valid str that still evaluates to true.
        self.det_power = float(det_power) if det_power and det_power not in self.from_sample.na_values else None
        self.power_color = 'white' if not self.det_power else 'rgb({0},{0},{0})'.format(
            int(255 * self.det_power))  # takes floor of the value

        ## TODO: remove this if chain
        # if mut_category:
        #    if 'Missense_Mutation' in mut_category:
        #        self.color = 'blue'
        #    elif 'Silent' in mut_category:
        #        self.color = 'orange'
        #    elif 'Nonsense_Mutation' in mut_category:
        #        self.color = 'red'
        #    elif 'Splice' in mut_category:
        #        self.color = 'purple'
        ##

        # Output of CCF clustering
        self.cluster_assignment = cluster_assignment  # moved to ND VirtualEvent
        # self.post_CCF_1d = None

        # if local_cn_a1 is not None:
        #    self.local_cn_a1 = float(local_cn_a1)
        # else:
        #    self.local_cn_a1 = np.nan
        # if local_cn_a2 is not None:
        #    self.local_cn_a2 = float(local_cn_a2)
        # else:
        #    self.local_cn_a2 = np.nan

    # except:
    #	err = str(sys.exc_info()[1])
    #	raise Exception('\n'.join(['Incorrect input format for variant ', err, 'please review input data!']))  # TODO:Print variant
    #	# This error should no longer come up that often.

    @property
    def var_str(self):
        return self._var_str

    @property
    def ccf_grid_size(self):  # Return the grid size of the CCF.
        return len(self.ccf_1d)

    # TODO make sure things can't be one at time changed from outside class

    @property
    def maf_annotation(self):
        raise NotImplementedError  # not implemented yet

    @classmethod  # a class factory helper to build from different inputs
    def from_list(cls, li, from_sample=None, ccfgrid_size=101):
        ccf_list = li[
                   4:ccfgrid_size + 4]  # skip first 4 element 'Chromosome','Start_position','Reference_Allele','Tumor_Seq_Allele2'
        variant = li[:4]
        in_list = variant + [ccf_list] + li[ccfgrid_size + 4:]
        return cls(*in_list, from_sample=from_sample)

    @classmethod  # a class factory helper to build from different inputs
    def from_dict(cls, required_li, optional_parm_dict, from_sample=None, ccfgrid_size=101):
        optional_parm_dict['from_sample'] = from_sample
        return cls(*required_li, **optional_parm_dict)

    @classmethod  # a class factory helper to build from different inputs
    def from_som_mutation_zero(cls, som_mut, from_sample=None, ccfgrid_size=101):
        ccf_1d = ['1.00'] + ['0.00'] * (ccfgrid_size - 1)  # assume ccf=0
        return cls(som_mut.chrN, som_mut.pos, som_mut.ref, som_mut.alt, ccf_1d, ref_cnt=som_mut.ref_cnt, alt_cnt=0,
                   gene=som_mut.gene, prot_change=som_mut.prot_change, mut_category=som_mut.mut_category,
                   from_sample=from_sample)

    # a method to define == operator to compare mutations , i.e. Mut1==Mut2 if their variant strings equal
    # TODO be careful to avoid bugs when a specifc sample is required (e.g. alt/ref data)
    def __eq__(self, other):
        if other is None:
            return False
        if self.var_str == other.var_str:
            return True
        else:
            return False

    # hash method to define mutation equality in sets, dictionaries etc.
    def __hash__(self):
        return hash(self.var_str)

    def __str__(self):
        return self.var_str

    def __repr__(self):
        # print "str_called"
        return self.var_str

    def clean_local_cn(self, cn1, cn2):
        self.local_cn_a1 = float(cn1) if not np.nan else np.nan
        self.local_cn_a2 = float(cn2) if not np.nan else np.nan

    # update allelic copy number AND assignment of mutation to a copy number event (arm level)
    def _phase_mutation(self, bam_file):
        # TODO: Phase mutation to an allele based on shifts of local hetsites and local copy number
        raise NotImplementedError

    # @classmethod #a class factory helper to build from different inputs
    def _from_clustered_maf(cls):
        raise NotImplementedError

    # @classmethod #a class factory helper to build from different inputs
    def _from_absolute_RData(cls):
        # TODO: implement function

        raise NotImplementedError


##########################################################
# CopyNumberEvent() - class to store each CN event
##########################################################
class CopyNumberEvent():
    """CLASS info

        FUNCTIONS:

            public:

            private:

        PROPERTIES:

            regular variables:

            @property methods: none
    """
    ref = '-'
    alt = '-'
    ref_cnt = ''
    alt_cnt = ''
    prot_change = ''

    def __init__(self, chrN, cn_category, start=0, end=0, arm=None, ccf_1d=None, ccf_hat=None, ccf_high=None, ccf_low=None,
                 std=None, from_sample=None, seg_tree=None, clust_ccf=None, local_cn=np.nan, a1=True,
                dupe=False):

        # try:
        # Link to sample object.
        self.from_sample = from_sample

        # Basic properties of mutation
        self.chrN = chrN
        self.start = start
        self.end = end
        self.arm = arm
        if ccf_1d:
            self.ccf_1d = ccf_1d
        else:
            std = std if std is not None else max((ccf_high - ccf_low) / 4., .001)
            if ccf_hat <= std:
                self.ccf_1d = np.insert(np.zeros(100), 0, 1.)
            elif ccf_hat >= 1. - std:
                self.ccf_1d = np.append(np.zeros(100), 1.)
            else:
                alpha = ccf_hat * ccf_hat * ((1 - ccf_hat) / (std * std) - (1 / ccf_hat))
                beta = alpha * ((1 / ccf_hat) - 1)
                if alpha < 1.:
                    self.ccf_1d = np.insert(np.zeros(100), 0, 1.)
                elif beta < 1.:
                    self.ccf_1d = np.append(np.zeros(100), 1.)
                else:
                    self.ccf_1d = scipy.stats.beta.pdf(np.arange(101.) / 100, alpha, beta)
                    self.ccf_1d /= sum(self.ccf_1d)
        self.seg_tree = seg_tree
        self.clust_ccf = clust_ccf

        self.type = 'CNV'

        # self._var_str = ':'.join(map(str, (cn_category, chrN, start.band, end.band, 'a1' if a1 else 'a2')))
        #
        # self.event_name = cn_category + str(chrN) + start.band + '-' + end.band[1:] if start != end else mut_category \
        #                                                                                                   + str(
        #     chrN) + start.band
        if cn_category.startswith('Arm'):
            gl = cn_category.split('_')[1]
            self.event_name = gl + '_' + str(self.chrN) + self.arm
        elif cn_category.startswith('Focal'):
            gl = cn_category.split('_')[1]
            self.event_name = gl + '_' + str(chrN) + start.band + '-' + end.band[1:] if start != end else gl + str(chrN) + start.band
        elif cn_category == 'WGD':
            self.event_name = 'WGD'
        else:
            raise ValueError('Invalid cn category provided: "{}"'.format(cn_category))
        self.event_name += '_' if dupe else ''
        self._var_str = self.event_name

        self.cluster_assignment = None
        if a1:
            self.local_cn_a1 = local_cn
            self.local_cn_a2 = np.nan
        else:
            self.local_cn_a2 = local_cn
            self.local_cn_a1 = np.nan
        self.a1 = a1
        self.cn_category = cn_category
        # self.set_mut_category(mut_category, arm=arm)

    def __hash__(self):
        return hash(self._var_str)

    def __str__(self):
        return self._var_str

    def __repr__(self):
        return self.event_name

    def __len__(self):
        return self.end - self.start + 1

    @property
    def var_str(self):
        return self._var_str

    # def set_mut_category(self, mut_category, arm=None,
    #                      cytoband=os.path.dirname(__file__) + '/supplement_data/cytoBand.txt'):
    #     self.mut_category = mut_category
    #     if mut_category == 'WGD':
    #         self.gene = 'WGD'
    #     elif mut_category == 'Arm_loss':
    #         self.gene = 'loss_' + str(self.chrN) + arm
    #     elif mut_category == 'Arm_gain':
    #         self.gene = 'gain_' + str(self.chrN) + arm
    #     elif mut_category.startswith('Focal'):
    #         bands = []
    #         with open(cytoband, 'r') as f:
    #             for line in f:
    #                 row = line.strip('\n').split('\t')
    #                 if row[0].strip('chr') != str(self.chrN):
    #                     continue
    #                 if int(row[1]) < self.end and int(row[2]) > self.start:
    #                     bands.append(row[3])
    #                 if int(row[1]) > self.end:
    #                     break
    #         bands = sorted(bands)
    #         section = bands[0] + '-' + bands[-1] if len(bands) > 1 else bands[0]
    #         if mut_category.endswith('loss'):
    #             self.gene = 'loss_' + str(self.chrN) + section
    #         elif mut_category.endswith('gain'):
    #             self.gene = 'gain_' + str(self.chrN) + section


class SomMutationND:
    """CLASS info

        FUNCTIONS:

            public:

            private:

        PROPERTIES:

            regular variables:

            @property methods: none
    """

    def __init__(self, chrN, pos, ref, alt, ccf_1d, ref_cnt=None, alt_cnt=None, gene=None, prot_change=None,
                 mut_category=None, det_power=None, from_sample=None, type_=None, clonality=None, on_arm_event=None,
                 seg_tree=None, clust_ccf=None, multiplicity=np.nan, local_cn_a1=np.nan, local_cn_a2=np.nan,
                 type_local_cn_event=np.nan,
                 clonal_cut_off=0.86):

        # try:
        # Link to sample object.
        self.from_sample = from_sample

        # Basic properties of mutation
        self.chrN = chrN
        self.pos = int(float(pos))
        self.ref = ref
        self.alt = alt
        self.ref_cnt = int(
            float(ref_cnt)) if ref_cnt is not None else None  # TODO: add support/handling for text CCF files
        self.alt_cnt = int(float(alt_cnt)) if alt_cnt is not None else None
        self.gene = gene if gene is not None and 'unknown' not in gene.lower() else None

        if type_ is None:
            # guess type for backwards compatibility, but prefer that type be passed explicitly.
            if "-" in self.ref:
                self.type = Enums.MutType.INS
            elif "-" in self.alt:
                self.type = Enums.MutType.DEL
            else:
                self.type = Enums.MutType.SNV
        else:
            self.type = type_
        # variant string, set as private, gotten with self.var_str
        self._var_str = ":".join(map(str, [self.chrN, self.pos, self.ref, self.alt]))

        try:
            self.ccf_1d = tuple(map(float, ccf_1d))  # make sure all ccf values are floats
            # TODO: CCF broadening - this needs to be implemented. Currently does nothing.
            if max(self.ccf_1d) > 0.25:  # heuristic - if 25% of the weight is in one bin, broaden it.
                self.ccf_1d = self._broaden_ccf(self.ccf_1d)
        except:
            logging.debug("Cannot parse ccf_1d in SomMutation instantiation")
            self.ccf_1d = None

        self.blacklist_status = False
        self.graylist_status = False

        self.prot_change = prot_change if prot_change is not None and 'unknown' not in prot_change.lower() else None  # Unknown values are set as __UNKNOWN__ in ocotator, but Unknown is also something that should be skipped for gene names.

        # several values/placeholders for output options
        self.mut_category = mut_category

        # Output of CCF clustering
        self.cluster_assignment = None
        self.post_CCF = None


##############################################################
# CN_SegProfile() - class to store copy number profile
##############################################################
class CN_SegProfile:
    """CLASS info

        FUNCTIONS:

            public: None

            private:
                _auto_file_type -- autodetect input data type
                _load_segs -- load segments and call _results function
                _merge_segments_in_segtree -- helper function when reading in segments
                _results_from_seg_file -- parser from seg_file
                _results_from_RData -- parser from Absolute RData
                _results_from_db -- parser from segments from sqlite3 db

        PROPERTIES:

            regular variables:
                seg_tree -- Interval Tree per chromosome formed from input CN profile
                merged_seg_tree -- version of seg_tree with merged adjacent segments of same allelic copy number

            @property methods: none
    """

    # csize contains chromosome bp lengths
    CSIZE = [0, 249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431,
             135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,
             63025520, 48129895, 51304566, 156040895, 57227415]

    # centromeres (define arm-level lengths)
    CENT_LOOKUP = {1: 125000000, 2: 93300000, 3: 91000000, 4: 50400000, 5: 48400000,
                   6: 61000000, 7: 59900000, 8: 45600000, 9: 49000000, 10: 40200000,
                   11: 53700000, 12: 35800000, 13: 17900000, 14: 17600000, 15: 19000000,
                   16: 36600000, 17: 24000000, 18: 17200000, 19: 26500000, 20: 27500000, 21: 13200000,
                   22: 14700000, 23: 60600000, 24: 12500000}

    def __init__(self, seg_file, input_type='auto', from_sample=None):
        # try: # validate correct input
        self.chroms = list(map(str, range(1, 23)))
        self.chroms.append('X')
        self.chroms.append('Y')
        self.from_sample = from_sample  # from_sample can be an Object, and potential to get purity from this
        logging.debug('reading in :' + seg_file)
        if input_type == 'auto':
            seg_file_type = self._auto_file_type(seg_file)
            logging.debug('found seg file type:' + seg_file_type)
        else:
            seg_file_type = input_type
        self.seg_tree = {x: IntervalTree() for x in self.chroms}
        if seg_file_type in ['tab', 'absolute', 'sqlite']: self._load_segs(seg_file, seg_file_type)
        logging.debug(self.seg_tree)
        ## overwrite merged_seg_tree with normal seg tree (raw seg tree)
        self.merged_seg_tree = self.seg_tree

        '''
        except:
            print "BAD SEG FILE:", seg_file
            #err = str(sys.exc_info()[1])
            #raise Exception('\n'.join(['Incorrect input format for seg file ',err,'please review input data!']))
        '''

    def chrom2int(self, chrom):
        try:
            return int(chrom)
        except:
            if chrom == 'X':
                return 23
            else:
                return 24

    # =======input file readers=======#
    def _auto_file_type(self, seg_file):  # func private to the class

        file_extension = os.path.splitext(seg_file)[1]
        if file_extension in ['.txt', '.tsv']:
            return 'tab'
        elif file_extension == '.RData':
            return 'absolute'
        elif file_extension == '.db':
            return 'sqlite'
        else:
            print >> sys.stderr, "ERROR: Cannot guess file type please use .RData, .txt, .tsv or .db"
        sys.exit(1)

    def _load_segs(self, seg_file, input_type='absolute'):

        if input_type == 'tab':
            # determine which headers to use for a1 and a2 -- expecting either absolut pre DP or post DP
            with open(seg_file, 'r') as segs:
                for i, row in enumerate(segs):
                    spl = row.strip("\n").split("\t")
                    if i == 0:
                        h = {x[1].lower(): x[0] for x in enumerate(spl)}
                        if "a1_cn" in h.keys():
                            file_type = "simulated"
                            header_indeces = [h.get("chromosome"), h.get("a1_cn"), h.get("a2_cn"), h.get('a1_cn'),
                                              h.get('start_position'), h.get('end_position')]
                        elif "a1.seg.cn" in h.keys():
                            file_type = "post_DP_segfile"
                            header_indeces = [h.get("chromosome"), h.get("a1.seg.cn"), h.get("a2.seg.cn"),
                                              h.get("a1.sigma"), h.get("start"), h.get("end")]
                        elif "star" in h.keys():
                            file_type = "PCAWG_consensus"
                            header_indeces = [h.get("chromosome"), h.get("minor_cn"), h.get("major_cn"), h.get("star"),
                                              h.get("start"), h.get("end"), h.get("absolute_broad_major_cn"),
                                              h.get("absolute_broad_minor_cn")]
                        else:
                            file_type = "pre_DP_segfile"
                            header_indeces = [h.get("chromosome"), h.get("rescaled.cn.a1"), h.get("rescaled.cn.a2"),
                                              h.get("seg_sigma"), h.get("start.bp"), h.get("end.bp")]
                        break
            return self._results_from_seg_file(seg_file, header_indeces, file_type)

        elif input_type == 'absolute':
            self.seg_tree = self.from_sample._results_from_RData(seg_file)

        elif input_type == 'sqlite':
            return self._results_from_db()

    # @classmethod #a class factory helper to build from different inputs
    def _results_from_seg_file(self, seg_file, header_indeces, file_type):

        seg_tree = {}
        for chrom in map(str, CN_SegProfile.CENT_LOOKUP.keys()):
            if chrom == '23': seg_tree['X'] = IntervalTree()
            if chrom == '24':
                seg_tree['Y'] = IntervalTree()
            else:
                seg_tree[chrom] = IntervalTree()
        if file_type == "PCAWG_consensus":
            use_star = True
            [chrom_idx, a1_idx, a2_idx, sigma_idx, start_idx, end_idx, broad_major_cn_idx,
             broad_minor_cn_idx] = header_indeces
        else:
            use_star = False
            [chrom_idx, a1_idx, a2_idx, sigma_idx, start_idx, end_idx] = header_indeces
        with open(seg_file, 'r') as segs:
            for i, row in enumerate(segs):
                spl = row.strip("\n").split("\t")
                if i == 0: continue
                if spl[a1_idx] == "NA" or spl[a2_idx] == "NA" or np.math.isnan(float(spl[a1_idx])) or np.math.isnan(
                        float(spl[a2_idx])):
                    continue
                else:
                    if use_star:
                        # if spl[sigma_idx] not in []:
                        if spl[sigma_idx] not in ['2', '3']:
                            # use subclonal segs from absolute
                            if spl[broad_minor_cn_idx] == "NA" or spl[broad_major_cn_idx] == "NA": continue
                            min_a_cn = float(spl[broad_minor_cn_idx])
                            maj_a_cn = float(spl[broad_major_cn_idx])
                        else:
                            min_a_cn = float(spl[a1_idx])
                            maj_a_cn = float(spl[a2_idx])
                    else:
                        min_a_cn = float(spl[a1_idx])
                        maj_a_cn = float(spl[a2_idx])

                    clonal_cn_min_a = round(min_a_cn)
                    clonal_cn_maj_a = round(maj_a_cn)
                    subc_cn_maj = clonal_cn_maj_a + np.sign(maj_a_cn - clonal_cn_maj_a)
                    subc_cn_min = clonal_cn_min_a + np.sign(min_a_cn - clonal_cn_min_a)
                    if subc_cn_maj == clonal_cn_maj_a:
                        sub_f_maj = 0
                    else:
                        sub_f_maj = abs((maj_a_cn - clonal_cn_maj_a) / float(subc_cn_maj - clonal_cn_maj_a))
                    if subc_cn_min == clonal_cn_min_a:
                        sub_f_min = 0
                    else:
                        sub_f_min = abs((min_a_cn - clonal_cn_min_a) / float(subc_cn_min - clonal_cn_min_a))

                    seg_tree[str(spl[chrom_idx])][int(float(spl[start_idx])):int(float(spl[end_idx]))] = \
                        [min_a_cn, maj_a_cn, clonal_cn_maj_a, clonal_cn_min_a, sub_f_maj, sub_f_min, subc_cn_maj,
                         subc_cn_min]

        self.seg_tree = seg_tree

    # @classmethod #a class factory helper to build from different inputs
    def _results_from_db(self):
        # TODO
        pass


##############################################################
# EventPair() - class to store event pairs for relative timing
##############################################################
########### Needed for backwards compatability ###############
class Event_Pair():

    def __init__(self, event1, event2, WGD_status, from_sample=None):
        self.from_sample = from_sample
        self.event1 = event1
        if event1.type == 'CNV':
            self.event1_type = 'CN_Event'
        else:
            self.event1_type = 'Mutation'
        self.event2 = event2
        if event2.type == 'CNV':
            self.event2_type = 'CN_Event'
        else:
            self.event2_type = 'Mutation'
        self.WGD_status = WGD_status
        self.winner = None
        self.reason = None
        self.prob_event1_before_event2 = None
        self.prob_event2_before_event1 = None
        self.prob_event_ordering_unknown = None
        self._hash = hash(":".join([event1.gene, str(event1.pos), event2.gene, str(event2.pos)]))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other

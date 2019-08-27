'''
Sample class to load and store data for each sample of the individual
'''

import os
import sys
import logging
import collections
import numpy as np
from intervaltree import Interval, IntervalTree

from scipy.interpolate import interp1d

from SomaticEvents import SomMutation, CopyNumberEvent

na_values = {'-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A', 'N/A', 'NA', '#NA', 'NULL', 'NaN', '-NaN', 'nan',
             '-nan', ''}  # http://pandas.pydata.org/pandas-docs/stable/io.html#na-values

class TumorSample:
    """CLASS info

      FUNCTIONS:
          public:

            add_sample() - add new samples from individual to self.sample_list by making a new TumorSample instance

         private:

           _load_sample_ccf() - guess input file type for ccf data if not manually provided

      PROPERTIES:
          regular variables:
             self.
              sample_list

          @property methods: temporarily_removed - all mutations that are graylisted but not blacklisted in this sample
    """

    common_field_map = {"Hugo_Symbol": "gene", "t_alt_count": "alt_cnt", "observed_alt": "alt_cnt", "ref": "ref_cnt",
                        "t_ref_count": "ref_cnt", "Variant_Classification": "mut_category",
                        "Protein_Change": "prot_change", "Variant_Type": "type_"}  # For reading TSV files.

    def __init__(self, file_name, input_type,
                 indiv=None,
                 sample_name=None, sample_short_id=None,
                 DNAsource=None,
                 timepoint=None,
                 ccf_grid_size=101,
                 artifact_blacklist=os.path.join(os.path.dirname(__file__), 'supplement/Blacklist_SNVs.txt'),
                 artifact_whitelist='',
                 PoN=None,
                 use_indels=False,
                 min_coverage=8,
                 delete_auto_bl=False,
                 _additional_muts=None,
                 seg_file=None,
                 purity=None, timepoint_value=None,
                 seg_input_type='auto'):

        self.indiv = indiv  ## $Patient object
        self.sample_name = sample_name
        self.sample_short_id=sample_short_id
        self.DNAsource = DNAsource
        self.ccf_grid_size = ccf_grid_size
        self.timepoint = timepoint_value
        # self._CN_Events = {}  # "11":[["11q22",102708080,119059462]],"14":[["test",106369481,106369485]]} #this is in development, currently does nothing. Meant to be events to look for.

        self._mut_varstring_hashtable = {}  # a dictionary hash table for fast var_str lookup
        self.private_mutations = []
        self.artifacts_in_blacklist = []
        with open(artifact_blacklist, 'r') as f:
            self.known_blacklisted_mut = {m.replace('\t', ':').strip() for m in f}
        if os.path.exists(artifact_whitelist):
            with open(artifact_whitelist) as f:
                self._whitelist = {m.replace('\t', ':').strip() for m in f}
        else:
            self._whitelist = None

        # Both are a dict for easy lookup later.
        self.low_coverage_mutations = {}


        # load and filter mutations
        self.mutations = self._load_sample_ccf(file_name, input_type, min_coverage=min_coverage, use_indels=use_indels,
                                               _additional_muts=_additional_muts)  # a list of SomMutation objects

        #TODO
        self.CnProfile = self._resolve_CnEvents(seg_file, input_type=seg_input_type)

        # @results
        self.ClustersPostMarginal = None  # format F[Cluster] = CCF post hist

        self.concordant_variants = []  # store variants concordant with other tumor samples
        # self.concordant_with_samples = []  # store  with other tumor samples were used for variants concordance



    @staticmethod
    def _rebin_to_n(y, n):
        f = interp1d(np.linspace(0, 1, len(y)), y)
        return (list(f(np.linspace(0, 1, n))))

    @property
    def temporarily_removed(self):
        return set(self.low_coverage_mutations.values())

    # TODO: limit ability of people to change mut from outside of the class
    @property  # dynamically set variable (if mutations are updated)
    def mut_varstr(self):
        return [x.var_str for x in self.mutations + self.low_coverage_mutations.values()]

    def get_mut_by_varstr(self, variant_string):
        return self._mut_varstring_hashtable[variant_string]

    # \=======function handles loading of CCF data for the sample========================
    def _load_sample_ccf(self, filen, input_type='absolute', min_coverage=8, use_indels=False,
                         _additional_muts=None):
        # input_type abs; txt; sqlite3 .db # auto tab if .txt, .tsv or .tab ; abs if .Rdata; sqlite if .db
        # ccf_dat is a dictionary 'variant_string': (ccf tuple)

        #autodetect file type
        input_type=self._auto_file_type(filen)


        mut_with_ccf_dat = {}

        if input_type == 'absolute':  # absolute
            mut_with_ccf_dat = self._read_absolute_results(filen)
        elif input_type == 'tab':
            mut_with_ccf_dat = self._read_ccf_from_txt(filen)
        elif input_type == 'calc_ccf':  # when only abs CN and ref/alt counts present
            raise NotImplementedError("Please provide precomputed raw ccf values!")
            ##cn_tree = self._read_CN_from_rdata(filen)
        elif input_type == 'sqlite':
            mut_with_ccf_dat = self._read_ccf_from_sqlite(filen)
        else:
            logging.error(
                "Unknown file type specified. Supported: Absolute .RData results, tab.delim txt file with CCF ")
            sys.exit(1)

        # check that grid size corresponds to data size
        # and then remove blacklisted mutations
        if _additional_muts is not None and input_type != "calc_ccf":  # if calc_ccf, don't add muts twice.
            ##TODO:Expermintal function to read in mutations from second file. Not done yet.
            logging.info("loading added X chromosome mutations!")
            mut_with_ccf_dat.extend(self._read_ccf_from_txt_default_headers(_additional_muts, _only_x=True))

        filtered_mutations = []
        for mut in mut_with_ccf_dat:
            # TODO: Fix this if loop to be more elegant

            self._mut_varstring_hashtable[mut.var_str] = mut  # All mutations, including filtered mutations should be included. in hashtable

            if mut.ccf_grid_size != self.ccf_grid_size:
                logging.debug("Rebinning mutation from {} to {} bins".format(mut.ccf_grid_size, self.ccf_grid_size))

                mut.ccf_1d = self._rebin_to_n(mut.ccf_1d, self.ccf_grid_size)
                mut.ccf_grid_size = self.ccf_grid_size

            if mut.var_str in self.known_blacklisted_mut:
                logging.info("Removed mut " + mut.var_str + " in " + self.sample_name)
                mut.blacklist_status = True

            print mut.gene, mut.prot_change, ";",
            if any([np.isnan(x) for x in mut.ccf_1d]):
                logging.warning("NAN Mutation - " + str(mut.gene) + "," + mut.var_str)
                self.known_blacklisted_mut.add(mut.var_str)
                mut.blacklist_status = True

            if mut.alt_cnt is not None and mut.alt_cnt + mut.ref_cnt < min_coverage:
                mut.graylist_status = True
            if (mut.type in ["INS", "DEL"]) and not use_indels:
                mut.graylist_status = True

            if mut.type == "CNV":
                mut.graylist_status = True
            # mut.ccf_1d=np.sqrt(np.sqrt(np.sqrt(mut.ccf_1d)))
            # if np.argmax(mut.ccf_1d) != 100: mut.graylist_status = True

            if self._whitelist is not None:
                if mut.var_str not in self._whitelist:
                    self.artifacts_in_blacklist.append(mut)
                    self.known_blacklisted_mut.add(mut.var_str)
                    continue

            if mut.blacklist_status:
                self.artifacts_in_blacklist.append(mut)  # TODO: make this a property that is auto computed from all graylisted/whitelisted mutations.
                continue
            if mut.graylist_status:
                self.low_coverage_mutations[mut.var_str] = mut
                continue

            filtered_mutations.append(mut)

        mut_with_ccf_dat = filtered_mutations
        print
        logging.info("Present in blacklist: {}".format(len(self.artifacts_in_blacklist)))
        logging.debug(",".join(map(str, self.artifacts_in_blacklist)))

        return mut_with_ccf_dat


    # \=======input file readers========================

    def _read_absolute_results(self, filen):  # func private to the class #reads .Rdata files from R

        # excute util script to extract data from .Rdata files and then load results a txt
        logging.info("Extracting from {0} file to {0}.ccf.tsv".format(filen))
        import subprocess
        subprocess.check_call(["Rscript", os.path.join(os.path.dirname(__file__), '../utils/rdata_extractor.R'), filen])
        return self._read_ccf_from_txt(filen + ".ccf.tsv")

    def _read_ccf_from_txt(self, filen, calc_ccf=False, _only_x=False, cn_tree=None):  # func private to the class
        # Only x paramter is to load X chr mutations from a different data source. Avoid this unless you know you need it.
        import scipy.interpolate

        file_in = open(filen) if type(filen) == str else filen  # allow buffers.

        header = file_in.readline()
        while header[0] == "#" or not header.strip():
            header = header.readline()
        header = header.strip().split("\t")
        h = collections.OrderedDict([[x[1], x[0]] for x in enumerate(header)])

        def get_cel_prev(mult, allele_cn, allele_ccf, other_cn, other_cn_ccf, alt, ref, PURITY, ncn):
            # PURITY = 0.5 # purity of sample
            # now given all of above model the CCF distributions
            # get the predicted AF
            # PURITY=0.9
            grid_size = float(101)
            x = np.linspace(0, 1, grid_size)
            real_af_hist = []
            a = alt + 1
            b = ref + 1
            af_hist = scipy.stats.beta.pdf(x, a, b) / (grid_size - 1)
            ccf_hist = []
            _ccf = x * (
                    allele_cn * allele_ccf * PURITY + other_cn * other_cn_ccf * PURITY + (1 - other_cn_ccf) * PURITY + (
                    1 - allele_ccf) * PURITY + ncn * (1 - PURITY)) / (
                           mult * PURITY - x * (PURITY * mult + (allele_cn - mult) * PURITY - allele_cn * PURITY))
            f = scipy.interpolate.interp1d(_ccf, af_hist)
            normal_dist = f(x)  # /sum(f(x))
            # plt.plot(x,normal_dist,'r',linewidth=2)
            return normal_dist

        mutation_list = []

        #check location of required columns

        #find ccf_bin locations
        ccf_bins_location = [column_loc[1] for column_loc in h.iteritems() if "ccf_raw_" in column_loc[0]]  # needs change if other header names

        #if "ccf_raw_" not in file look for oldstyle headers "0", "0.01", "1"
        if len(ccf_bins_location) ==0:
            ccf_bin_cols = ['0'] + map(str,[round(0.01*i,2) for i in range(1,100)]) + ['1']
            ccf_bins_location = [ column_loc[1] for column_loc in h.iteritems() if column_loc[0] in ccf_bin_cols ]

        #another format
        if len(ccf_bins_location) == 0:
            ccf_bins_location = [column_loc[1] for column_loc in h.iteritems() if "ccf_0." in column_loc[0] or "ccf_1." in column_loc[0]]

        for line in file_in:
            spl = line.strip().split('\t')
            if "Chromosome" in h:
                std_param = [spl[h["Chromosome"]], spl[h["Start_position"]], spl[h["Reference_Allele"]],
                             spl[h["Tumor_Seq_Allele2"]]]
            else:
                # assume that first 4 'Chromosome','Start_position','Reference_Allele','Tumor_Seq_Allele2'
                std_param = spl[:4]
                ccf = spl[4:4 + self.ccf_grid_size]

            if calc_ccf:
                ccf = [float("NaN")] * self.ccf_grid_size  # calculate after creation.
            else:
                try:
                    ccf = [float(spl[x]) for x in ccf_bins_location]  # assume ccf at the end of the split since headers vary for this one.
                except ValueError:
                    logging.warning('Mutation with no CCF estimate... skipping')
                    continue

            #print len(ccf)
            if len(ccf)!=101: raise ValueError("Number of CCF bins values read for this variant are less than 101 bins !")
            # add ccf to std list
            std_param.append(ccf)

            # header_optional_keys=["gene","prot_change"]
            # header_optional_keys = header[len(std_param) + len(ccf) - 1:]  # keys have to match the ones in SomMutation
            opt_dict = {}
            for key in self.common_field_map.keys():
                if key in h:
                    opt_dict[self.common_field_map[key]] = spl[h[key]]

            if _only_x and spl[h["Chromosome"]] != "X": continue
            mut = SomMutation.from_dict(std_param, opt_dict, from_sample=self)

            if calc_ccf:
                try:
                    maj_a, min_a, frac_s_maj, frac_s_min, nMaj2_A, nMin2_A = \
                        list(cn_tree[str(mut.chrN)][mut.pos:mut.pos + 1])[0].data
                    mut.ccf_1d = ccf_hist.get_ccf(maj_a, min_a, frac_s_maj, frac_s_min, nMaj2_A, nMin2_A, mut.alt_cnt,
                                                  mut.ref_cnt, cn_tree["purity"])

                except IndexError:
                    logging.warning("Mutation doesn't overlap CN! " + str(mut))
                except ValueError:
                    logging.error(
                        map(str, [maj_a, min_a, frac_s_maj, frac_s_min, nMaj2_A, nMin2_A, mut.alt_cnt, mut.ref_cnt]))

            mutation_list.append(mut)

        # the tabfile should have 4 first columns for variant then ccf columns
        file_in.close()

        # remove duplicate variants from list:
        dupl_removed = list(collections.OrderedDict.fromkeys(mutation_list))
        if len(dupl_removed) != len(mutation_list):
            # warining message if duplicates found
            sample_name = self.sample_name if self.sample_name is not None else ''
            logging.warning("Duplicate mutations found in sample", sample_name + ":")
            logging.warning("Removed", len(mutation_list) - len(dupl_removed), "mutations.")
            mutation_list = dupl_removed

        # print a.columns
        # mutation_list.sort(key=lambda x: x.var_str) if sorting is required
        return mutation_list

    def _read_ccf_from_vcf(self, filen):
        raise NotImplementedError

    def _read_ccf_from_sqlite(self, filen):
        raise NotImplementedError

    def _results_from_RData(self, seg_file):
        sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
        import utils.rdata_loader
        R_path = os.path.join(os.path.dirname(__file__), "../utils")
        for file in os.listdir(R_path):
            if file.endswith(".R"):
                source_file = os.path.join(R_path, file)
                logging.info("Sourcing " + source_file)
                pyR.r.source(source_file)
        purity, seg_tree = utils.rdata_loader.load_abs_rdata(seg_file)
        self.purity = purity
        return seg_tree

    def _get_local_cn_for_each_mut(self):
        logging.info("Getting local cn for each mutation")
        for mut in set(self.mutations).union(set(self.low_coverage_mutations.values())):
            # update local cn
            if len(self.seg_profile.merged_seg_tree[mut.chrN][mut.pos]) > 0:
                cn1, cn2 = list(self.seg_profile.merged_seg_tree[mut.chrN][mut.pos])[0].data[0:2]
                mut.clean_local_cn(cn1, cn2)

    # ======small class utils====================
    # @staticmethod
    def _auto_file_type(self, filen):  # func private to the class

        file_extension = os.path.splitext(filen)[1]

        if file_extension in ['.txt', '.tab', '.tsv']:
            return 'tab'
        elif file_extension == '.RData':
            return 'absolute'
        elif file_extension == '.db':
            return 'sqlite'
        else:
            raise IOError("ERROR: Cannot guess file type please use .RData, .txt, .tsv, .tab or .db")

    def add_muts_to_hashtable(self, muts):
        if hasattr(muts, '__iter__'):
            self._mut_varstring_hashtable.update((mut.var_str, mut) for mut in muts)
        else:
            self._mut_varstring_hashtable[muts.var_str] = muts

    def _resolve_CnEvents(self, seg_file, input_type='auto'):
        if input_type == 'auto':
            if not seg_file:
                input_type = 'none'
            elif seg_file.endswith('.segtab.txt') or seg_file.endswith('.seg.txt'):
                input_type = 'absolute'
            elif seg_file.endswith('.tsv'):
                input_type = 'alleliccapseg'
            else:
                input_type = 'moo'

        seg_tree = {chrom: IntervalTree() for chrom in list(map(str, range(1, 23)))+['X', 'Y']}

        if input_type == 'none':
            return None
        elif input_type == 'absolute':
            with open(seg_file, 'r') as fh:
                header = fh.readline().strip('\n').split('\t')
                for line in fh:
                    try:
                        row = dict(zip(header, line.strip('\n').split('\t')))
                        chrN = row['Chromosome']
                        start = int(row['Start.bp'])
                        end = int(row['End.bp'])
                        ccf_hat_a1 = float(row['cancer.cell.frac.a1'])
                        ccf_high_a1 = float(row['ccf.ci95.high.a1'])
                        ccf_low_a1 = float(row['ccf.ci95.low.a1'])
                        ccf_hat_a2 = float(row['cancer.cell.frac.a2'])
                        ccf_high_a2 = float(row['ccf.ci95.high.a2'])
                        ccf_low_a2 = float(row['ccf.ci95.low.a2'])
                        local_cn_a1 = int(row['modal.a1'])
                        local_cn_a2 = int(row['modal.a2'])
                        seg_tree[chrN].add(Interval(start, end,
                            (self.sample_name, {'cn_a1': local_cn_a1, 'cn_a2': local_cn_a2, 'ccf_hat_a1': ccf_hat_a1,
                                'ccf_high_a1': ccf_high_a1, 'ccf_low_a1': ccf_low_a1, 'ccf_hat_a2': ccf_hat_a2,
                                'ccf_high_a2': ccf_high_a2, 'ccf_low_a2': ccf_low_a2})))
                    except ValueError:
                        continue
        else:
            raise NotImplementedError('Can only read absolute segtab file')

        return seg_tree
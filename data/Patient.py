##########################################################
## Patient - central class to handle and store sample CCF data
##########################################################
from Sample import TumorSample
import os
import logging
import itertools
import numpy as np

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
        self.PatientLevel_MutBlacklist = None
        self.PatientLevel_MutWhitelist = None

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
        self.sample_list
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


##############################################################

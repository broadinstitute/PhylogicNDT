# PhylogicNDT

**#This is Python2 version branch, please use "python3" branch for Python3 update
**
## Installation 
First:  Clone this repository

    git clone https://github.com/broadinstitute/PhylogicNDT.git
    cd PhylogicNDT

Then either :
	
### Manual Install 
Install python 2.7, R (optional) and required packages 
For debian:

    apt-get install python-pip build-essential python-dev r-base r-base-dev git graphviz libgraphviz-dev


Install setuptools and wheel

	pip install setuptools wheel
	
Install required packages 

	pip install -r req
	
Install scipy, matplotlib, and pandas (these versions are recommended) 

	pip install pandas==0.19.2 scipy==1.0.0 matplotlib==2.0.0
	pip install -e git+https://github.com/rmcgibbo/logsumexp.git#egg=sselogsumexp (for faster compute)




### Docker Install
Install docker from https://www.docker.com/community-edition#/download

	docker build --tag phylogicndt . 

## Using the Package

    ./PhylogicNDT.py --help

If running from the docker, first run:

	docker run -i -t phylogicndt
	cd phylogicndt


### Clustering 

To run clustering on the provided sample input data:

 To specify inputs: 

	./PhylogicNDT.py Cluster -i Patient_ID  -s Sample1_id:Sample1_maf:Sample1_CN_seg:Sample1_Purity:Sample1_Timepoint -s Sample2_id:Sample2_maf:Sample2_CN_seg:Sample2_Purity:Sample2_Timepoint ... SampleN_info 

alternatively - provide a tsv sample_information_file (.sif) 

with headers: sample_id maf_fn seg_fn purity timepoint

    ./PhylogicNDT.py Cluster -i Patient_ID  -sif Patient.sif

the .maf should contain pre-computed raw ccf histograms based on mutations alt/ref count 
(Absolute annotated mafs or .Rdata files are also supported)
if the ccf histograms are absent - the `--maf_input_type` flag must be set to
`calc_ccf` and sample purity must be provided. Also local copy number must
be attached to each mutation in the maf with columns named `local_cn_a1` and `local_cn_a2`

CN_seg is optional to annotate copy-number information on the trees

To specify number of iterations: 

	./PhylogicNDT.py Cluster -ni 1000

	
<sub>Acknowledgment: Clustering Module is partially inspired (primary 1D clustering) by earlier work of Carter & Getz (Landau D, Carter S , Stojanov P et al. Cell 152, 714–726, 2013)</sub>
	
### BuildTree (and GrowthKinetics) 
The GrowthKinetics module fully incorporates the BuildTree libraries, so when rates are desired, there is no need to run both. 

 - The -w flag should provide a measure of tumor burden, with one value per input sample maf in clustering. **When ommited, stable tumor burden is assumed.** 
  - The -t flag should provide relative time for spacing the samples.
    **When omitted, equal spacing is assumed.** 

Just BuildTree

	./PhylogicNDT.py BuildTree -i Indiv_ID -sif Patient.sif  -m mutation_ccf_file -c cluster_ccf_file 

GrowthKinetics

	./PhylogicNDT.py GrowthKinetics -i Indiv_ID -sif Patient.sif -ab cell_population_abundance_mcmc_trace -w 10 10 10 10 10 -t 1 2 3 4 5 

Run Cluster together with BuildTree

	./PhylogicNDT.py Cluster -i Patient_ID  -sif Patient.sif -rb

### SinglePatientTiming

SinglePatientTiming requires a maf input and a seg file input for each sample.
The maf file should be the output of PhylogicNDT Clustering module.
The seg file should have the following columns:

    Chromosome  Start   End A1.Seg.CN   A2.Seg.CN

To run SinglePatientTiming:

    ./PhylogicNDT.py Timing -i Indiv_ID -sif Patient.sif

### LeagueModel
LeagueModel requires an input of comparison tables.
The comparison tables should be the output of SinglePatientTiming ending in ".comp.tsv"

To run LeagueModel:

    ./PhylogicNDT.py LeagueModel -cohort Cohort -comps comp1 comp2 ... compN

Alternatively, one can use a single aggregated table. The table should have the following columns:

    sample  event1  event2  p_event1_win    p_event2_win    unknown

To run with the aggregated table:

    ./PhylogicNDT.py LeagueModel -cohort Cohort -comparison_cn comps

### PhylogicSim 
A simulation module is provided for convenience.

    ./PhylogicNDT.py PhylogicSim --help

Command to visualize all the options and help.

    ./PhylogicNDT.py PhylogicSim 

Run the simulation with the default paramters.

    ./PhylogicNDT.py PhylogicSim -i MySimulation

Specify a prefix for all the output files

    ./PhylogicNDT.py PhylogicSim -i MySimulation -ns 7

Specify the number of samples you want to simulate. 

    ./PhylogicNDT.py PhylogicSim -i MySimulation -nodes 5

Specify the number of distinct clones present in your samples. Minimum 2 (The first clone is always the clonal clone)

    ./PhylogicNDT.py PhylogicSim -i MySimulation -nodes 5 -seg /Example_SegFile.txt

Specify a segment file with copy number values to sample from. See the "Example_SegFile.txt" for a format example. If no file is specified, a build-in CN profile is used, based on the hg19 contigs.

    ./PhylogicNDT.py PhylogicSim -i MySimulation -nodes 5 -clust_file /Example_Clust_File.txt

Force the ccf values of each cluster on each sample, instead of generating a new random phylogeny from scratch. If -clust_file is specified, the -ns and -nodes flags are ignored an instead replaced with the values from the Clust_File. Each line of the tsv file represents a sample, with each tab separated value the ccf of a cluster. The last value of each line must always be -1 to account for the artifact cluster. 

    ./PhylogicNDT.py PhylogicSim -i MySimulation -nodes 5 -clust_file /Example_Clust_File.txt -a 0.3

Specify the proportion of mutations that are artifactual (Random af unrelated to mutation/CN). Can be combined with a clust_file.

    ./PhylogicNDT.py PhylogicSim -i MySimulation -nodes 5 -clust_file /Example_Clust_File.txt -pfile /Example_PurityFile.txt

TSV file to specify the purity of each sample individualy (Otherwise, the purity is specified for all the samples using the -p flag.). Each line represents a sample. The file can optionally contain an extra three columns with the alpha, beta and N values for the coverage betabinomial for each sample (Otherwise, those values are set for all samples using the -ap, -b and -nb flags respectively). 

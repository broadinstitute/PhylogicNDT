from bitnami/minideb
RUN install_packages python3-pip build-essential python-dev-is-python3 r-base r-base-dev git graphviz python3-tk
RUN pip install setuptools wheel
RUN pip install numpy scipy matplotlib pandas
COPY req /tmp/req
RUN apt-get -y upgrade
RUN apt-get -y update
RUN apt-get install -y libgraphviz-dev
RUN pip install -r /tmp/req
RUN pip install -e git+https://github.com/rmcgibbo/logsumexp.git#egg=sselogsumexp
RUN mkdir /phylogicndt/
COPY PhylogicSim /phylogicndt/PhylogicSim
COPY GrowthKinetics /phylogicndt/GrowthKinetics
COPY BuildTree /phylogicndt/BuildTree
COPY Cluster /phylogicndt/Cluster
COPY SinglePatientTiming /phylogicndt/SinglePatientTiming
COPY LeagueModel /phylogicndt/LeagueModel
COPY data /phylogicndt/data
COPY ExampleData /phylogicndt/ExampleData
COPY ExampleRuns /phylogicndt/ExampleRuns
COPY output /phylogicndt/output
COPY utils /phylogicndt/utils
COPY PhylogicNDT.py /phylogicndt/PhylogicNDT.py
COPY LICENSE /phylogicndt/LICENSE
COPY req /phylogicndt/req
COPY README.md /phylogicndt/README.md

# Loading Initiation Elongation Termination (LIET) Model
## Table of contents
1. [LIET model description] (## LIET model description)
2. [Installation] (## Installation)
3. [Usage] (## Usage)
4. [Example output] (## Example output)
5. [Contact information] (## Contact information)

## LIET model description 

The LIET model parameterizes the transcription process, by annotating nascent run-on sequencing data. Below are the parameters LIET outputs on a per-gene basis from nascent run-on sequencing data: 

*$\mu_L$: Polymerase loading position 
*$\mu_L'$: Polymerase antisense loading position 
*$\sigma_L$: Polymerase loading uncertainty
*$\sigma_L'$: Polymerase antisense loading uncertainty
*$T_{I}$: Polymerase initiation length 
*$T_{I}'$: Polymerase antisense initiation length
*$\mu_T$: Position of polymerase dissociation
*$\sigma_T$: Uncertainty of polymerase dissociation
*$W_{LI}$: Proportion of reads that fall into loading and initiation
*$W_{E}$: Proportion of reads that fall into elongation
*$W_{T}$: Proportion of reads that fall into termination 
*$W_{B}$: Proportion of reads that fall into background signal

![The LIET Model](./README-figs/LIET_MAIN_FIG1.pdf)

## Installation 
1. To install LIET, first set up Miniconda on the cluster you are running LIET on. Instructions on how to do this are on the anaconda website [here][https://docs.anaconda.com/miniconda/]. Once you are on the website, scroll to the "quick command line install" and run the appropriate command based on your operating system (eg: macOS, Linux, or Windows). 

2. After you have installed Miniconda, install the PyMC (version 5.6) package into a new conda environment. 
```
conda create -c conda-forge -n your_env_name "pymc==5"
conda activate your_env_name
```
*Note, do not specifiy the last digit of PyMC version 5.6 (eg: do not write `pymc==5.6` in the command above)*

3. Check the version of PyMC you have installed by running the command below. You should have installed PyMC version 5.6.
```
conda list
```

4. When you run LIET, make sure to activate the PyMC environment you have just created using the following command:
```
conda activate your_env_name
```

5. Once PyMC is installed, clone the LIET GitHub repository. 

## Usage 

### Config file
LIET takes one config file per sample run, which points the software to where the annotation file (what genes you want LIET to annotate), bedgraph files (input data), pad file (how many extra bases you want to add to the 5' and 3' ends of the gene annotation supplied in the annotation file to ensure LIET captures the entire transcription process), and path to where you want your results located/your result file name. 

An example configuration file is shown below (*Note example priors are set in the configuration file below, however these should be tailored to your gene set and data LIET is being run on*). The `[MODEL]`, `[DATA_PROC]`, `[FIT]`, and `[RESULTS]` sections of the config file do not need to change for basic LIET usage on nascent run-on sequencing data.
```
[FILES]
ANNOTATION=/path-to-your-annotation-file/
BEDGRAPH_POS=/path-to-your-positive-bedgraph-file/
BEDGRAPH_NEG=/path-to-your-negative-bedgraph-file/
RESULTS=/path-to-your-results-directory/result-file-name.liet
PAD_FILE=/path-to-your-pad-file/

[MODEL]
ANTISENSE=True
BACKGROUND=True
FRACPRIORS=False

# Example [PRIORS] below. 
# These priors are a good starting place for running LIET.
# Reccomendation: play with changing prior values & look at associated LIET fits.
# This will help you determine which values are the best for your genes/data. 
# More information about priors can be found on PyMC's website.
[PRIORS]
mL=dist:normal,mu:0,sigma:1500
sL=dist:exponential,offset:1,tau:500
tI=dist:exponential,offset:1,tau:500
mT=dist:exponential,offset:1,tau:10000
sT=dist:exponential,offset:10,tau:500
w=dist:dirichlet,alpha_LI:1,alpha_E:1,alpha_T:1,alpha_B:1
mL_a=dist:normal,mu:0,sigma:1500
sL_a=dist:exponential,offset:1,tau:500
tI_a=dist:exponential,offset:1,tau:500

[DATA_PROC]
RANGE_SHIFT=True
PAD=10000,30000

[FIT]
ITERATIONS=50000
LEARNING_RATE=0.05
METHOD=advi
OPTIMIZER=adamax
MEANFIELD=True
TOLERANCE=1e-5

[RESULTS]
SAMPLES=50000
MEAN=True
MODE=False
MEDIAN=False
STDEV=True
SKEW=False
PDF=False
```
### Annotation file
The `ANNOTATION` file (specified in the config file) is a tab separated/headerless file that contans the set of genes you want to run LIET on. The format for the annotation file is `chromosome start stop gene length-of-gene strand`

Example annotation file: 
```
chr1	6185019	6199595	RPL22	14576	-
chr1	8860999	8878686	ENO	17687	-
chr1	10032957	10181239	UBE4B	148282	+
chr1	11054588	11060018	SRM	-
chr1	15409887	15430339	EFHD2	20452	+
chr1	15617457	15669044	DDI2	51587	+
chr1	17018751	17054032	SDHB	35281	-
chr1	19882394	19912945	OTUD3	30551	+
chr1	20499447	20508151	MUL1	8704	-
chr1	21440136	21484900	NBPF3	44764	+

```
### Bedgraph files
`BEDGRAPH_POS` & `BEDGRAPH_NEG` (specified in the config file) are the input data for LIET. These files should be strand separated bedgraph files. Because the LIET model assumes each read is a direct measure of polymerase activity it is important that the bedgraph files **only** contain the 3-prime-most position of ech nascent read. Therefore, each read fed into LIET should only be one base long. If you supply the LIET model with entire reads, the software will take **significantly** longer. 

### Pad file
The `PAD_FILE` (specified in the config file) is a tab separated/headerless file that contains the padding regions you want to add on to your annotated genes. This feature gives LIET more space to search for each parameter, given transcription begins before the annotated transcription start site and ends after the RNA cleavage site. The format for the annotation file is `gene 5’pad,3’pad`.

Example pad file: 
```
RPL22	5000, 25000
ENO1	10000, 30000
UBE4B	10000, 25000
SRM	3500, 7000
EFHD2	10000, 25000
DDI2	9000, 11000
SDHB	9000, 6000
OTUD3	3000, 30000
MUL1	10000, 9000
NBPF3	7500, 16000
```
It is important to note that **a pad file is not necessary to run LIET**. A defult pad is supplied in the `[DATA_PROC]` section of the config file. The following `[DATA_PROC]` example  would set default pads of 10,000 bases added onto the 5' end and 30,000 bases onto the 3' end at every gene if no pad file was provided. 

```
[DATA_PROC]
RANGE_SHIFT=True
PAD=10000,30000
```
## Example output
After LIET successfully runs, it will output 3 files: `your_output_name.liet`, `your_output_name.liet.log`, and  `your_output_name.liet.err`

### `your_output_name.liet` file description 
The `your_output_name.liet` file contains the model output in the form of `param=value;stdev-of-value` for `mL` ($\mu_L$), `sL` ($\sigma_L$), `mL_a` ($\mu_L'$), `sL_a` ($\sigma_L'$), `tI` ($T_{I}$), `tI_a` ($T_{I}'$),`mT` ($\mu_T$), and `sT` ($\sigma_T$). 

This file also contains the relative propertions of each read in each stage of transcription ("weights"). Weights for each parameter are also stored in this file, in the format of `w=[weight-of-loading-and-initiation, weight-of-elongation, wieght-of-termination, weight-of-background]` for the sense strand, and `w_a=[weight-of-antisense-loading-and-initiation, weight-of-antisense-background]` for the antisense strand. This file also contains the annotation information associated with each gene that was properly fit, eg: `chr start stop strand gene`.

*Note all output parameters are reported in units of bases relative to the input annotation `start` position*

### `your_output_name.liet.log` description 
The `your_output_name.liet.log` file contains ancillary information for each gene from each fit. This file contains the range of bases (referenced from the input annotation `start` position the model used to fit each gene (`fit_range`), the coverage on each strand (`strand_cov`), the minimum residual value and residual value in the first iteration of fitting (`elbo_range`), the number of iterations LIET fit the data (`iterations`), and how long the model took to fit each gene in units of minutes (`fit_time_min`). 

### `your_output_name.liet.err` description 
The `your_output_name.liet.log` file contains information about any errors that occured when fitting each gene. If no errors occured, the file will just write the name of the gene that was fit to this file. 

### Plotting your fits
After running LIET, **assessment of the output fits are essential** to ensure the model worked as anticipated. The `liet_res_class.py`, `rnap_lib_fitting_results.py` and `rnap_lib_plotting.py` libraries are helpful for plotting model fits. 

Example workflow for plotting fits: 
1. Import necessary packages 

![Import-pacakages-plotting.png](./README-figs/Import-pacakages-plotting.png)


2. Define LIET output files and pull out model fits for specified genes

![Define-input-gene-fit-info.png](./README-figs/Define-input-gene-fit-info.png)


3. Plot LIET fit


![plot-LIET.png](./README-figs/plot-LIET.png)


## Contact information
jacob.stanley@colorado.edu
georgia.barone@colorado.edu






# Resources for running LIET

## From the Original LIET paper

This directory contains the pad, annotation, and `#SBATCH` files used in the LIET paper [1]. Below are descriptions of what each file is and how it should be used: 

* **chr1-6-3p-UTR.liet.ann** is a LIET annotation file containing genomic coordinates for isolated genes on chromosomes 1-6. This annotation file contains the inferred poly-A sites (3' end of the UTR for each isoform selected) for each isolated gene. *This annotation file is how we selected poly-A sites in the LIET paper.*
* **chr1-6.liet.ann** is a LIET annotation file that also contains genomic coordinates for isolated genes on chromosomes 1-6. Unlike the previous file, this one does not include poly-A sites. Instead, it references the 5' end of the UTR or the last exon, based on a visual assessment of the fit. This approach allows the model more flexibility in searching for mT. *This annotation file was the annotation file fed to LIET for the analysis presented in the LIET paper.*
* **chr1-6.pad** is a LIET pad file that contains the padding regions for the isolated gene set used in the analysis presented in the LIET paper. This file should be used in conjunction with *chr1-6.liet.ann*, as the padding regions are specifically tailored to the 3' end site selected (either the 5' end of the UTR or the last exon) found in chr1-6.liet.ann.
* **example-submission.sbatch** is an example `#SBATCH` script with information on how to submit LIET jobs to the computing cluster. 

## For running on enhancer RNAs
### 1. Prepping files to run LIET
* **01_run_LIET_prep_annsplit.sh** is the bash script that runs LIET_prep_annsplit.sh with your different inputs as described below
* **LIET_prep_annsplit.sh** is a bash script that optimizes creating the annotation files, config files, and sbatch scripts for LIET if you are running a large set of regions (e.g. 1k+ over multiple ssamples). Although the default config parameters are optimized for enhancers, this same script can be used with genes. In general it does the following
        
    1) Splits annotation files up so pymc use of pytensor cache does not fail (will occur with too many regions done at once) and separates regions that are overlapping 
        
    2) Writes the config files, scripts, etc so you can just do *sbatch X.sbatch* to run a sample
    * **Inputs**
        1) SRR_list: Name of file with prefixes of samples to be used (one prefix per row)
        2) Annfile: LIET Annotation file for all regions of interest (e.g. chr1-6.liet.ann))
        3) Input_dir: directory with bedgraphs or bam files -- format listed in 5)
        4) Output Directory: directory that will store the final annotation files, configs, sbatch scripts, and results
        5) Input Type: 3 inputs are allowed:
            * BAMS: Assumes file format of ${SRR}.sorted.bam. Will filter these bams for multi-mapped reads and convert to 3' bedgraphs
            * CLEAN_BAMS: Assumes file format of ${SRR}.mmfilt.sorted.bam
            * BGS: Assumes file format of 3' bedgraphs with name ${SRR}.sorted_3.bedGraph (same as those used for Tfit from [Bidirectional-Flow](https://github.com/Dowell-Lab/Bidirectional-Flow/tree/Tfit_focus))
            * BGS_done: Already stranded 3' bedgraphs in file format of ${SRR}_3.pos.sorted.bedGraph and ${SRR}_3.pos.sorted.bedGraph.
        6) Ann_Prefix: The prefix used for the split annotation files (e.g. P53_Enhancers_3.12)
        7) PAD: The pad to be used:
            * General pad of 3000bp both sides = "PAD=3000,3000"
            * Using a pad file = "PAD=${PAD_FILE}"

    * **Output** that will appear in the set Output Directory
        * e_and_o/: where all of the error and out files will be stored from sbatch runs
        * mapped/: IF you input anything but BGS_done, the created 3' stranded bedgraphs are stored here
        * LIET/
            * ANN_DIR_TEMP/: Where the split up annotation files are (format of [0-9]_${prefix}.txt)
            * configs/:
                * Every sample gets its own directory (e.g. Sample1) that then has configs corresponding to every split up annotation file (format of [0-9]_${prefix}_EMG.liet.config)
            * run-liet/:
                * sbatch script to run each sample.
            * LIET_results/:
                * Every sample gets its own directory that will store the final results after you run the sbatch scripts to run LIET (empty after only running run_LIET_prep_annsplit.sh)
                * The output from running the sbatch script will be separated by each split up annotation file BUT collected together to a final output with ${SRR}_EMG.liet[.err|.log]



### 1A. Run LIET
* Actually run the sbatch scripts under run-liet (e.g. ```sbatch test1.sbatch```)

### 2. Check for errors due to pytensor caching
Pymc relies on pytensor caching which is NOT optimized for large repetitive tasks. Despite the edits made for prep, models occassionally fail solely due to pytensor caching. We created the following pipeline to easily rerun these cases:

* 02_rerun_LIET_errors.sh
    * **Parameters to edit:**
        * **OD**: Same as Output Directory used in 01_run_LIET_prep_annsplit.sh
        * **LIET_DIR**: Same as LIET Directory used in 01_run_LIET_prep_annsplit.sh
        * **PAD**: Same as PAD used in 01_run_LIET_prep_annsplit.sh
        * **BGS**: the directory where the stranded 3' bedgraphs are used (if used 'BGS_done' option then same as 01_run_LIET_prep_annsplit.sh, otherwise ${OD}/mapped/bedgraphs)
        * **srr_list**: The list of Samples you want to consider (like ones in file used for 01_run_LIET_prep_annsplit.sh )

* **New Output** that will appear in the Output Directory
    * LIET/
        * ANN_DIR_TEMP: If the sample had errors, there will be new split up annotation files with the names [0-9]_redo_annot_${SRR}_EMG.txt
        * configs: Within each sample's directory, there is a new inner directory called redo_errors that stores the config files for the regions that errored
        * run_liet: The sbatch scripts for just the errored regions will be named as ${SRR}_EMG_error.sbatch
        * LIET_Results: Each sample will get their own error focused directory named as ${SRR}_EMG_err

### 3. Getting the Final Stranded Bed files based on LIET
* A jupyter notebook showing how to get the final regions in the form of stranded bed graphs from LIET, both with consensus regions and individual samples, can be found at 03_Get_LIET_results_bed.ipynb


**References**
1. [LIET Model: Capturing the kinetics of RNA polymerase from loading to termination](https://www.biorxiv.org/content/10.1101/2024.10.03.616401v1)

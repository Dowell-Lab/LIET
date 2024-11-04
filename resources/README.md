# LIET annotation, pad, and `#SBATCH` files

This directory contains the pad, annotation, and `#SBATCH` files used in the LIET paper [1]. Below are descriptions of what each file is and how it should be used: 

* **chr1-6-3p-UTR.liet.ann** is a LIET annotation file containing genomic coordinates for isolated genes on chromosomes 1-6. This annotation file contains the inferred poly-A sites (3' end of the UTR for each isoform selected) for each isolated gene. *This annotation file is how we selected poly-A sites in the LIET paper.*
* **chr1-6.liet.ann** is a LIET annotation file that also contains genomic coordinates for isolated genes on chromosomes 1-6. Unlike the previous file, this one does not include poly-A sites. Instead, it references the 5' end of the UTR or the last exon, based on a visual assessment of the fit. This approach allows the model more flexibility in searching for mT. *This annotation file was the annotation file fed to LIET for the analysis presented in the LIET paper.*
* **chr1-6.pad** is a LIET pad file that contains the padding regions for the isolated gene set used in the analysis presented in the LIET paper. This file should be used in conjunction with *chr1-6.liet.ann*, as the padding regions are specifically tailored to the 3' end site selected (either the 5' end of the UTR or the last exon) found in chr1-6.liet.ann.
* **example-submission.sbatch** is an example `#SBATCH` script with information on how to submit LIET jobs to the computing cluster. 

**References**
1. [LIET Model: Capturing the kinetics of RNA polymerase from loading to termination](https://www.biorxiv.org/content/10.1101/2024.10.03.616401v1)

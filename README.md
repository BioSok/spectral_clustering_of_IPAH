## Spectral Clustering of IPAH
This repository contains scripts of the R code used to generate the results in our paper "Biological heterogeneity in idiopathic pulmonary arterial hypertension identified through unsupervised transcriptomic profiling of whole blood".

## System requirements
* An Intel-compatible platform running Windows 10 /8.1/8 /7 /Vista /XP /2000 Windows Server 2019 /2016 /2012 /2008 /2003
* At least 256 MB of RAM, a mouse, and enough disk space for recovered files, image files, etc.
* The administrative privileges are required to install and run R‑Studio utilities.
* A network connection for data recovering over network.
* Tested on RStudio, Version 1.2.5042


## How to run

##### Run pre_clustering_dataset.R
This step includes the specific preprocessing for our paper "Biological heterogeneity in idiopathic pulmonary arterial hypertension identified through unsupervised transcriptomic profiling of whole blood".

Inputs:  
- RNA-sequencing file (genes x patients) : `rnaseq_data.xlsx`
- Clinical variable file (patients x variables) : `clinical_data.xlsx`

Outputs:  
- Pre clustering ready file : `pre_clustering_p_all_tpm.RDS`

>Run-time for 300 genes and 359 patients: <= 10 seconds

##### Run p_clustering.R
For thi step you can use the demo dataset `demo_pre_clustering_p_all_tpm.RDS` and gene list `demo_sorted_variant_genes.RDS`, provided in this repository.

Inputs:  
- Pre clustering ready file : `pre_clustering_p_all_tpm.RDS`
- Number of subgroups (k)
- Number of most variant genes to be used: (top_genes)

Outputs:
- Gene list sorted based on variance(descending order) : `sorted_variant_genes.RDS`
- Subgroup memberships for patients : `memberships_k5.RDS` & `memberships_k5.csv`

>Run-time for 300 genes and 359 patients: <= 1 min

## Contact
Please contact Sokratis Kariotis (Biosok) through Github for queries relating to this code.

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Spectral Clustering of IPAH
This repository contains scripts of the R code used to generate the results in our paper "Biological heterogeneity in idiopathic pulmonary arterial hypertension identified through unsupervised transcriptomic profiling of whole blood".


Biological heterogeneity in idiopathic pulmonary arterial hypertension identified through unsupervised transcriptomic profiling of whole blood

## System requirements

## How to run

##### Run pre_clustering_dataset.R
Inputs:  
- RNA-sequencing file (genes x patients) : rnaseq_data
- Clinical variable file (patients x variables) : clinical_data.xlsx

Outputs:  
- Pre clustering ready file : pre_clustering_p_all_tpm.RDS
  
##### Run p_clustering.R

Inputs:  
- Pre clustering ready file : pre_clustering_p_all_tpm.RDS
- Gene list sorted based on variance(descending order) : sorted_variant_genes.RDS

Outputs:  
- Subgroup memberships for patients : memberships_k5.RDS & memberships_k5.csv

## Contact

Please contact Sokratis Kariotis (Biosok) through Github for queries relating to this code.

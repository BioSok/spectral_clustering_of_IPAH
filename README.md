# Spectral Clustering of IPAH
Biological heterogeneity in idiopathic pulmonary arterial hypertension identified through unsupervised transcriptomic profiling of whole blood

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
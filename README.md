# spectral_clustering_of_IPAH
Biological heterogeneity in idiopathic pulmonary arterial hypertension identified through unsupervised transcriptomic profiling of whole blood

1. Run pre_clustering_dataset.R
  
  Inputs:
      (I) RNA-sequencing file (genes x patients) : 
      (II) Clinical variable file (patients x variables) : clinical_data.xlsx
      
  Outputs:
      (I) Pre clustering ready file : pre_clustering_p_all_tpm.RDS
  

2. Run p_clustering.R
      
  Inputs:
      (I) Pre clustering ready file : pre_clustering_p_all_tpm.RDS
      (II) Gene list sorted based on variance(descending order) : sorted_variant_genes.RDS
      
  Outputs:
      (I) Subgroup memberships for patients : memberships_k5.RDS & memberships_k5.csv
  
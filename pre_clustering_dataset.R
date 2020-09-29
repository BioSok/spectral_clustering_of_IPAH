source("internal_scripts/sample_exclusions.R")

# Initial datasets
rnaseq_file <- "rnaseq_data"
clinical_file <- "clinical_data"

# Parameters
clust_type <- "p"
top_genes <- 300
remove_duplicated_samples = FALSE # FALSE because these are different rows in the clinical file (may be removed later)
remove_column_samples = TRUE # TRUE because it removes non-IPAH and samples with duplicate RNA-seq rows 
remove_low_rin = -1
remove_PVODs = TRUE
rna_normalisation = "asinh"
remove_bias_genes <- c("PRKY", "TTTY15", "AC006032.1", "RPS4Y1", 
                       "EIF1AY", "KDM5D", "TXLNG2P", "USP9Y", "ZFY", "DDX3Y", "UTY" ) # male genes to be removed

datasets_to_use <- sample_exclusions(rnaseq_file, 
                                     clinical_file, 
                                     remove_duplicated_samples, 
                                     remove_column_samples,
                                     remove_low_rin, 
                                     remove_PVODs, 
                                     rna_normalisation)

updated_rnaseq_file <- datasets_to_use[[1]]
updated_clinical_file <- datasets_to_use[[2]]

# Remove biased genes
if(length(remove_bias_genes) != 0) {
    updated_rnaseq_file <- updated_rnaseq_file[!rownames(updated_rnaseq_file) %in% remove_bias_genes,]
}

if(clust_type == "p") {
    # ==== Retaining of patients ====
    p_df <- filter(updated_clinical_file, sampleClass == "patient")
    samples <- p_df$SampleID
    updated_rnaseq_file <- updated_rnaseq_file[,samples]
}

sorted_variant_genes <- readRDS("sorted_variant_genes.RDS")
genes_to_use <- sorted_variant_genes[1:top_genes,2]

updated_rnaseq_file <- updated_rnaseq_file[genes_to_use,]

save_name <- paste0("pre_clustering_", clust_type, "_all_tpm.RDS")
saveRDS(updated_rnaseq_file, file = save_name)

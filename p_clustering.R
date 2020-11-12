# --------------- Required info --------------
k <- 5 # number of subgroups
top_genes <- 100
reproduction_seed <- 71101

# --------- Required scripts loading ---------
library(fpc)
library(kernlab)
library(xlsx)

source("internal_scripts/sample_exclusions.R")
source("internal_scripts/stable_geneset.R")

# --------- Data loading ---------
tpm <- readRDS("demo_pre_clustering_p_all_tpm.RDS") # genes x samples
tpm <- t(tpm) # samples x genes

# --------- Selecting gene set ---------
tpm <- stable_geneset(tpm, top_genes)

# --------- Spectral clustering ---------
set.seed(reproduction_seed)
clustering_result <- specc(tpm, centers = k, kernel = "rbfdot")
PAH_clusters <- data.frame(SampleID = rownames(tpm), clusters = clustering_result@.Data)

# --------- Saving membership file ---------
name_rds <- paste0("memberships_k",k,".RDS")
name_csv <- paste0("memberships_k",k,".csv")

saveRDS(PAH_clusters, file = name_rds)
write.csv(PAH_clusters, file = name_csv, row.names=FALSE)

table(PAH_clusters$clusters) # Subgroup sizes

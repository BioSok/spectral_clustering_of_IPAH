# Author: Sokratis Kariotis
# Date: 21 May 2019
# Description: Function that filters out samples from further analysis for various reasons
# Arguments
# rnaseq_file: name of RNA-seq file (without suffix)
# clinical_file: name of clinical file (without suffix)
# remove_duplicated_samples: BOOLEAN
# remove_column_samples: BOOLEAN
# remove_low_rin: either a number >= 0 (effective threshold) or a negative number (no filtering based on rin). Default = 0
# rna_normalisation: the normalisation applied to the RNA-seq tpm data. Default = "asihn"

# Example run: sample_exclusions("mdc95_cts", "180621 - pheno all with lasso model-1_updated_508samples", remove_duplicated_samples = TRUE, remove_column_samples = TRUE, remove_low_rin = 0, remove_PVODs = TRUE)

sample_exclusions <- function(rnaseq_file,
                              clinical_file, 
                              remove_duplicated_samples = FALSE, 
                              remove_column_samples = TRUE,
                              remove_low_rin = 0,
                              remove_PVODs = TRUE,
                              rna_normalisation = "asinh") {
    
    suppressMessages(library(dplyr))
    
    # load dataset scripts
    source("internal_scripts/rna_preprocessing.R")
    source("internal_scripts/clinical_preprocessing.R")
    
    total_removed_samples <- 0
    
    all.tpm <- rna_preprocessing(rnaseq_file, rna_normalisation)
    clinical_matrix <- clinical_preprocessing(clinical_file)
    
    # remove_duplicated_samples == TRUE only first occurences of duplicate column names
    if (remove_duplicated_samples) {
        tpm_sam_names <- colnames(all.tpm)
        dupl_tpm_sam_names <- tpm_sam_names[duplicated(tpm_sam_names)]
        dts_msg <- paste0("Removing the first occurence of the following ", 
                          length(dupl_tpm_sam_names), 
                          " duplicated samples/columns from the RNA-seq data: ", 
                          toString(dupl_tpm_sam_names))
        print(dts_msg)
        print("=====")
        all.tpm <- all.tpm[, !duplicated(tpm_sam_names, fromLast = TRUE)] 
        
        cl_sam_names <- clinical_matrix$SampleID
        dupl_cl_samples <- cl_sam_names[duplicated(cl_sam_names)]
        dcs_msg <- paste0("Removing the first occurence of the following ", 
                          length(dupl_cl_samples), 
                          " duplicated samples/rows from the clinical data: ", 
                          toString(dupl_cl_samples))
        print(dcs_msg)
        print("=====")
        clinical_matrix <- clinical_matrix[!duplicated(cl_sam_names, 
                                                       fromLast = TRUE),] 
        total_removed_samples <- total_removed_samples + 
            length(dupl_tpm_sam_names) + 
            length(dupl_cl_samples)
    }
    
    if (remove_column_samples) {
        # The category of "ExcludeCriteria" column to be excluded is: 
        # Not IPAH, Relatives
        # repeat samples that have duplicate RNA-seq rows (105FNO, 7037_v1, HV168)

        notIPAHs <- filter(clinical_matrix, ExcludeCriteria == "Not IPAH")
        ni <- as.vector(notIPAHs$SampleID)
        relatives <- filter(clinical_matrix, ExcludeCriteria == "Relative")
        re <- as.vector(relatives$SampleID)
        dupls <- colnames(all.tpm)[duplicated(colnames(all.tpm), fromLast = TRUE)]
        du <- as.vector(dupls)
            
        rcsRel_msg <- paste0("Removing the following ", 
                            dim(relatives)[1], 
                          " samples from the RNA-seq data due to being relatives of irrelevant samples: ",
                          toString(re))
        print(rcsRel_msg)
        print("=====")
        
        rcsNI_msg <- paste0("Removing the following ", 
                            dim(notIPAHs)[1], 
                            " samples from the RNA-seq data due to Not IPAH diagnosis: ",
                            toString(ni))
        print(rcsNI_msg)
        print("=====")
        
        rcsDup_msg <- paste0("Removing the following ", 
                             length(dupls), 
                             " samples from the RNA-seq data due to being relatives of irrelevant samples: ",
                             toString(du))
        print(rcsDup_msg)
        print("=====")
        
        # Removing samples from RNA-seq data
        all.tpm <- all.tpm[,!colnames(all.tpm) %in% notIPAHs$SampleID]
        all.tpm <- all.tpm[,!colnames(all.tpm) %in% relatives$SampleID]
        all.tpm <- all.tpm[, !duplicated(colnames(all.tpm), fromLast = TRUE)] 
        
        total_removed_samples <- total_removed_samples + 
            dim(relatives)[1] +
            dim(notIPAHs)[1] +
            length(dupls)
    }
    
    if(remove_low_rin >= 0) {
        rin_threshold <- remove_low_rin
        low_rins <- filter(clinical_matrix, rin == rin_threshold)
        low_rins_ids <- low_rins$SampleID
        ts <- as.vector(low_rins$SampleID)
        cn <- colnames(all.tpm)
        rin_msg <- paste0("Removing the following ", 
                          length(ts), 
                          " samples from the clinical & RNA-seq data due to low quality(<=", 
                          rin_threshold, "): ",
                          toString(ts))
        print(rin_msg)
        print("=====")
        all.tpm <- all.tpm[,!(cn %in% ts)]
        total_removed_samples <- total_removed_samples + 
            length(ts)
    }
    
    if(remove_PVODs) {
        pvods_to_exclude <- unlist(as.list(
            subset(clinical_matrix, diagnosis_verified%in%c("PVOD/PCH"))["SampleID"]))
        pvods_msg <- paste0("Removing the following ", 
                          length(pvods_to_exclude), 
                          " samples from the clinical & RNA-seq data because they are characterised as PVODs: ",
                          toString(pvods_to_exclude))
        print(pvods_msg)
        print("=====")
        all.tpm <- all.tpm[,!colnames(all.tpm) %in% pvods_to_exclude]
        clinical_matrix <- clinical_matrix[!clinical_matrix$SampleID %in% pvods_to_exclude,]
        total_removed_samples <- total_removed_samples + 
            length(pvods_to_exclude)
    }
    
    # Keep only samples that are in both RNA-seq and clinical datasets
    clinical_samples <- clinical_matrix$SampleID
    tpm_samples <- colnames(all.tpm)
    common_samples <- intersect(clinical_samples,tpm_samples)
    
    if(length(clinical_samples) < length(tpm_samples)) {
        dif <- setdiff(tpm_samples,clinical_samples)
        #dif <- c()
        matching_msg <- paste0("Removing the following ", 
                               length(dif), 
                               " samples from the RNA-seq data because we don't have clinical data for them: ",
                               toString(dif))
    }
    else if (length(clinical_samples) > length(tpm_samples)) {
        dif <- setdiff(clinical_samples,tpm_samples)
        matching_msg <- paste0("Removing the following ", 
                               length(dif), 
                               " samples from the clinical data because we don't have RNA-seq data for them: ",
                               toString(dif))
    }
    else {
        matching_msg <- "RNA-seq and clinical datasets contain the same samples!"
    }
    
    print(matching_msg)
    print("=====")
    clinical_matrix <- clinical_matrix[clinical_matrix$SampleID %in% common_samples,]
    #all.tpm <- all.tpm[,common_samples]
    
    data <- list(all.tpm, clinical_matrix)
    print(paste0("Total samples excluded from RNA-seq: ", total_removed_samples))
    return(data)
}

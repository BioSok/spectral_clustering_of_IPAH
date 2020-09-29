# Author: Sokratis Kariotis
# Date: 21 May 2019
# Description: Function that normalises the RNA-seq data (hypebolic arcsine)
# Arguments
# sf: RNA-seq file name (no suffixes)
# normalisation: numerical normalisation to be applied on the RNA-seq data

# Example run: rna_preprocessing("rnaseq_data", "asinh")

rna_preprocessing <- function(sf, normalisation) {
    # Loading the RNA-seq dataset
    source_file <- paste0(sf,".RDS")
    all.tpm <- readRDS(source_file)
    
    if(normalisation == "asinh") {
        all.tpm <- asinh(all.tpm)
    }
    
    return(all.tpm)
}
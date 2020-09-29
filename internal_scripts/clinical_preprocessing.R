# Author: Sokratis Kariotis
# Date: 21 May 2019
# Description: Function that preprocesses the clinical data (add per case)
# Arguments
# sf: Clinical file name (no suffixes)

# Example run: clinical_preprocessing("clinical_data")

clinical_preprocessing <- function(sf) {
    library(readxl)
    
    # Loading the clinical dataset
    source_file <- paste0(sf,".xlsx")
    clinical_matrix <- read_excel(source_file)
    return(clinical_matrix)
}
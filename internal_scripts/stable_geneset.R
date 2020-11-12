stable_geneset <- function(tpm, top_number) {
    
    var_genes <- data.frame(apply(tpm, 2, var))
    colnames(var_genes) <- "variance"
    
    sorted_var_genes <- var_genes[order(var_genes$variance,decreasing = TRUE), ,drop = FALSE]
    sorted_var_genes$names <- rownames(sorted_var_genes)
    top_list <- head(sorted_var_genes$names, n = top_number)
    new_tpm <- tpm[,top_list]
    
    saveRDS(sorted_var_genes, "../demo_sorted_variant_genes.RDS")
    
    return(new_tpm)
}

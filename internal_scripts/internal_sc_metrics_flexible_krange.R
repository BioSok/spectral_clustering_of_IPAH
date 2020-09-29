internal_sc_metrics_flexible_krange <- function(df ,min_k ,max_k) {
    
    library(knitr)
    library(kableExtra)
    
    k_array <- sprintf("k%s",min_k:max_k)
    k_length <- length(k_array)
    scores <- matrix(, nrow = 15, ncol = k_length)
    colnames(scores) <- k_array
    rownames(scores) <- c("calinski_harabasz","dunn","pbm","tau","gamma",
                          "c_index","davies_bouldin","mcclain_rao","sd_dis",
                          "ray_turi","g_plus","silhouette","s_dbw",
                          "Compactness","Connectivity")
    
    clusters <- matrix(, nrow = dim(df)[1], ncol = k_length)
    colnames(clusters) <- k_array
    rownames(clusters) <- rownames(df)
    
    counter <- 1
    for(cur_k in min_k:max_k) {
        
        flag <- FALSE
        tryCatch({
            set.seed(71101)
            criteria_obj = dice(df, nk = cur_k, algorithms = c("sc"), evaluate = TRUE)
            flag <- TRUE
        }, error = function(e) {})
        
        if(flag) {
            obj_clusters <- (criteria_obj$clusters)[,1]
            
            if(cur_k == 2) indices <- criteria_obj$indices$ii$`2`
            else if (cur_k == 3) indices <- criteria_obj$indices$ii$`3`
            else if (cur_k == 4) indices <- criteria_obj$indices$ii$`4`
            else if (cur_k == 5) indices <- criteria_obj$indices$ii$`5`
            else if (cur_k == 6) indices <- criteria_obj$indices$ii$`6`
            else if (cur_k == 7) indices <- criteria_obj$indices$ii$`7`
            else if (cur_k == 8) indices <- criteria_obj$indices$ii$`8`
            else if (cur_k == 9) indices <- criteria_obj$indices$ii$`9`
            else if (cur_k == 10) indices <- criteria_obj$indices$ii$`10`
            
            ccol <- c(indices$calinski_harabasz[1],indices$dunn[1],
                      indices$pbm[1], indices$tau[1],
                      indices$gamma[1], indices$c_index[1],
                      indices$davies_bouldin[1], indices$mcclain_rao[1],
                      indices$sd_dis[1],indices$ray_turi[1],
                      indices$g_plus[1], indices$silhouette[1],
                      indices$s_dbw[1], indices$Compactness[1],
                      indices$Connectivity[1])
            
            scores[, counter] <- ccol
            clusters[, counter] <- obj_clusters
        } else {
            warning("Assigning zero scores, one or more clusters have no members!")
            scores[, counter] <- c(-10000,-10000,-10000,-10000,-10000,
                                   10000,10000,10000,10000,10000,10000,
                                   -10000,10000,-10000,-10000)
            clusters[, counter] <- rep(0, dim(df)[1])
        }
        counter <- counter + 1
    }
    
    best_cs <- c(which(scores[1,]==max(scores[1,])), # calinski_harabasz (max)
                 which(scores[2,]==max(scores[2,])), # dunn (max)
                 which(scores[3,]==max(scores[3,])), # pbm (max)
                 which(scores[4,]==max(scores[4,])), # tau (max)
                 which(scores[5,]==max(scores[5,])), # gamma (max)
                 which(scores[6,]==min(scores[6,])), # c_index (min)
                 which(scores[7,]==min(scores[7,])), # davies_bouldin (min)
                 which(scores[8,]==min(scores[8,])), # mcclain_rao (min)
                 which(scores[9,]==min(scores[9,])), # sd_dis (min)
                 which(scores[10,]==min(scores[10,])), # ray_turi (min)
                 which(scores[11,]==min(scores[11,])), # g_plus (min)
                 which(scores[12,]==max(scores[12,])), # silhouette (max)
                 which(scores[13,]==min(scores[13,])), # s_dbw (min)
                 which(scores[14,]==max(scores[14,])), # Compactness
                 which(scores[15,]==max(scores[15,]))) # Connectivity

    scores %>%
        kable(caption = "Internal Indexes") %>%
        kable_styling(bootstrap_options = c("striped", "hover", 
                                            "condensed", "responsive"),
                      font_size = 8,full_width = T)

    return(list(scores,clusters,best_cs))
}
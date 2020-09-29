external_sc_metric_flexible_krange <- function(df ,min_k ,max_k) {
    
    library(ClusterR)
    library(dplyr)
    library(aricode)
    
    k2 <- readRDS("patient_healthy results/Memberships/ph_memberships_k2.RDS")
    colnames(k2) <- c("SampleID","k2")
    k3 <- readRDS("patient_healthy results/Memberships/ph_memberships_k3.RDS")
    colnames(k3) <- c("SampleID","k3")
    k4 <- readRDS("patient_healthy results/Memberships/ph_memberships_k4.RDS")
    colnames(k4) <- c("SampleID","k4")
    k5 <- readRDS("patient_healthy results/Memberships/ph_memberships_k5.RDS")
    colnames(k5) <- c("SampleID","k5")
    k6 <- readRDS("patient_healthy results/Memberships/ph_memberships_k6.RDS")
    colnames(k6) <- c("SampleID","k6")
    k7 <- readRDS("patient_healthy results/Memberships/ph_memberships_k7.RDS")
    colnames(k7) <- c("SampleID","k7")
    k8 <- readRDS("patient_healthy results/Memberships/ph_memberships_k8.RDS")
    colnames(k8) <- c("SampleID","k8")
    k9 <- readRDS("patient_healthy results/Memberships/ph_memberships_k9.RDS")
    colnames(k9) <- c("SampleID","k9")
    
    ph_clinical <- readRDS("patient_healthy results/ph_total_clinical_k5.RDS")
    ph_clinical <- ph_clinical[k2$SampleID,]

    true_labels <- select(ph_clinical, SampleID, sampleClass)
    true_labels$sampleClass[which(true_labels$sampleClass=="patient")] <- 1
    true_labels$sampleClass[which(true_labels$sampleClass=="control")] <- 0
    true_labels$sampleClass <- as.numeric(true_labels$sampleClass)
    
    final <- left_join(true_labels, k2, by = "SampleID")
    final <- left_join(final, k3, by = "SampleID")
    final <- left_join(final, k4, by = "SampleID")
    final <- left_join(final, k5, by = "SampleID")
    final <- left_join(final, k6, by = "SampleID")
    final <- left_join(final, k7, by = "SampleID")
    final <- left_join(final, k8, by = "SampleID")
    final <- left_join(final, k9, by = "SampleID")
    final[199,2] <- 1
    
    # k2_nmi <- external_validation(final$sampleClass, final$k2, method = "nmi", summary_stats = TRUE)
    # k3_nmi <- external_validation(final$sampleClass, final$k3, method = "nmi", summary_stats = TRUE)
    # k4_nmi <- external_validation(final$sampleClass, final$k4, method = "nmi", summary_stats = TRUE)
    # k5_nmi <- external_validation(final$sampleClass, final$k5, method = "nmi", summary_stats = TRUE)
    # k6_nmi <- external_validation(final$sampleClass, final$k6, method = "nmi", summary_stats = TRUE)
    # k7_nmi <- external_validation(final$sampleClass, final$k7, method = "nmi", summary_stats = TRUE)
    # k8_nmi <- external_validation(final$sampleClass, final$k8, method = "nmi", summary_stats = TRUE)
    # k9_nmi <- external_validation(final$sampleClass, final$k9, method = "nmi", summary_stats = TRUE)
    
    
    k2_nmi_aricode <- format(round(NMI(final$sampleClass, final$k2), 3), nsmall = 3)
    k3_nmi_aricode <- format(round(NMI(final$sampleClass, final$k3), 3), nsmall = 3)
    k4_nmi_aricode <- format(round(NMI(final$sampleClass, final$k4), 3), nsmall = 3)
    k5_nmi_aricode <- format(round(NMI(final$sampleClass, final$k5), 3), nsmall = 3)
    k6_nmi_aricode <- format(round(NMI(final$sampleClass, final$k6), 3), nsmall = 3)
    k7_nmi_aricode <- format(round(NMI(final$sampleClass, final$k7), 3), nsmall = 3)
    k8_nmi_aricode <- format(round(NMI(final$sampleClass, final$k8), 3), nsmall = 3)
    k9_nmi_aricode <- format(round(NMI(final$sampleClass, final$k9), 3), nsmall = 3)
    
    nmis <- data.frame(NMI = c(k2_nmi_aricode, k3_nmi_aricode, k4_nmi_aricode, k5_nmi_aricode,
                        k6_nmi_aricode, k7_nmi_aricode, k8_nmi_aricode, k9_nmi_aricode),
                       k = c("k2","k3","k4","k5","k6","k7","k8","k9"))
    
    saveRDS(nmis, file = "patient_healthy results/nmi_scores_k.RDS")
    
    ggplot(data=nmis, aes(x=k, y=NMI)) +
        geom_bar(stat="identity", fill="steelblue")+
        geom_text(aes(label=NMI), vjust=1.6, color="white", size=3.5)+
        ggtitle("NMI of each k, based on the patient & control labels") +
        theme(axis.text=element_text(size=12),
              axis.title=element_text(size=12,face="bold"),
              axis.text.x = element_text(angle = 90, vjust = 0.5),
              plot.title = element_text(hjust = 0.5, size = 14,face="bold"),
              legend.position="none")
    
    

}
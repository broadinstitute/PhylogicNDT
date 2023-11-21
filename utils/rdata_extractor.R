args <- commandArgs(trailingOnly = TRUE)
load(args[1])
maf.data <- seg.obj$mut.cn.dat[c('Hugo_Symbol','Chromosome', 'Start_position', 'Reference_Allele', 'Tumor_Seq_Allele2','Variant_Classification','Variant_Type','ref','observed_alt','Protein_Change')]
maf.data[maf.data==""] = NA
ccf.data <- seg.obj$mode.res$SSNV.ccf.dens[1,,]
df <- cbind(maf.data,ccf.data)
write.table(df,file=paste0(args[1],".ccf.tsv"),quote=FALSE,na="NA",sep="\t",row.names=FALSE)
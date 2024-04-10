BiocManager::install(version="3.17")
install.packages("devtools")
devtools::install_github("cellgeni/sceasy")

BiocManager::install("sceasy")
BiocManager::install("SingleCellExperiment")
install.packages("Seurat")

library(Seurat)
library(sceasy)
library(SingleCellExperiment)
seurat_object <- readRDS("~/Desktop/ependymagliosis.rds")
sce_object <- as.SingleCellExperiment(seurat_object)
sceasy::convertFormat(sce_object, from="sce", to="anndata", outFile="ependymagliosis.h5ad")
py_install("anndata")
















library(Seurat)
library(ggplot2)
library(dplyr)
library(SeuratDisk)
Canteen_clean <- readRDS("~/Desktop/ependymagliosis.rds")
load("~/Desktop/ependymagliosis.rds")
Canteen_clean
#Canteen_clean$obj
#Canteen_clean$obj=UpdateSeuratObject(Canteen_clean$obj)   #the robj name will be shown here, e.g example_data                               #the robj name will be shown here, e.g example_data
SaveH5Seurat(Canteen_clean, filename = "epq.h5Seurat",misc=F)

Convert("eypq.h5Seurat", dest = "h5ad")
# Load the necessary libraries

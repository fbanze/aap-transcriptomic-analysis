library(limma)
library(stats)
library(edgeR)
library(dplyr)
library(stringr)


args <- commandArgs(trailingOnly = TRUE)

split_idx_train <- read.csv(args[1])
split_idx_test <- read.csv(args[2])
fold <- args[3]
model <- args[4]
outer <- args[5]
PROJECT_ROOT <- normalizePath(args[6], winslash = "/", mustWork = TRUE)

# modified remove batch effect https://github.com/gangwug/limma

removeBatchEffect_modified <- function(x,batch=NULL,batch2=NULL,covariates=NULL,design=NULL,group=NULL,...)

{
  #	Covariates to remove (batch effects)
  if(is.null(batch) && is.null(batch2) && is.null(covariates)) return(as.matrix(x))
  if(!is.null(batch)) {
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[,-1,drop=FALSE]
  }
  if(!is.null(batch2)) {
    batch2 <- as.factor(batch2)
    contrasts(batch2) <- contr.sum(levels(batch2))
    batch2 <- model.matrix(~batch2)[,-1,drop=FALSE]
  }
  if(!is.null(covariates)) {
    covariates <- as.matrix(covariates)
    covariates <- t(t(covariates) - colMeans(covariates))
  }
  X.batch <- cbind(batch,batch2,covariates)

  #	Covariates to keep (experimental conditions)
  if(!is.null(group)) {
    group <- as.factor(group)
    design <- model.matrix(~group)
  }

  #	Check design
  if(is.null(design)) {
    message("design matrix of interest not specified. Assuming a one-group experiment.")
    design <- matrix(1,ncol(x),1)
  }

  #	Fit combined linear model
  x <- as.matrix(x)
  fit <- lmFit(x,cbind(design,X.batch),...)

  #	Subtract batch effects adjusted for experimental conditions
  beta <- fit$coefficients[,-(1:ncol(design)),drop=FALSE]
  beta[is.na(beta)] <- 0
  return(list(
    corrected = x - beta %*% t(X.batch),
    beta = beta
  ))
}

aap_counts <- read.csv(file=file.path(PROJECT_ROOT, 'datasets', 'exp_mat_unscaled.csv'))
colData <- read.csv(file=file.path(PROJECT_ROOT, 'datasets', 'colData.csv'))

rownames(aap_counts) <- aap_counts$X
aap_counts$X <- NULL
rownames(colData) <- colData$X

aap_lcpm_train <- aap_counts[,split_idx_train$index+1]
aap_lcpm_test <- aap_counts[,split_idx_test$index+1]
colData_train <- colData[split_idx_train$index+1,]
colData_test <- colData[split_idx_test$index+1,]

batch_train <- factor(colData_train$country)
levels_batch <- levels(batch_train)
batch_test <- colData_test$country
batch_test <- factor(batch_test, levels = levels_batch)

design <- model.matrix(~ as.factor(colData_train$sex) + as.factor(colData_train$age))
batch_corrected_train <- removeBatchEffect_modified(as.matrix(aap_lcpm_train),
                                                    batch=batch_train,
                                                    design=design)

contrasts(batch_test) <- contr.sum(levels(batch_test))
batch_test <- model.matrix(~batch_test)[,-1,drop=FALSE]
x <- as.matrix(aap_lcpm_test)
beta <- batch_corrected_train$beta

#Fals fehlende Werte vorhanden sind
missing_in_test <- setdiff(colnames(batch_train), colnames(batch_test))
if(length(missing_in_test) > 0) {
  zero_mat <- matrix(
    0,
    nrow = nrow(batch_test),
    ncol = length(missing_in_test),
    dimnames = list(NULL, missing_in_test)
  )
  batch_test <- cbind(batch_test, zero_mat)
}

#richtige Reihenfolge der Gene
batch_test <- batch_test[, colnames(batch_train), drop = FALSE]
beta <- beta[, colnames(batch_train), drop = FALSE]

x <- as.matrix(aap_lcpm_test)

batch_corrected_test <- x - beta %*% t(batch_test)
if(outer == "True"){
  write.csv(batch_corrected_train$corrected, paste(PROJECT_ROOT,"/datasets/", model ,"/train_full_corrected_", fold , ".csv", sep = '')) 
} else {
  write.csv(batch_corrected_train$corrected, paste(PROJECT_ROOT,"/datasets/", model ,"/train_corrected_", fold , ".csv", sep = '')) 
}
mode <- str_detect(args[2], "val")
if(mode){
  write.csv(batch_corrected_test, paste(PROJECT_ROOT,"/datasets/", model, "/val_corrected_", fold , ".csv", sep = ''))
} else {
  write.csv(batch_corrected_test, paste(PROJECT_ROOT,"/datasets/", model, "/test_corrected_", fold , ".csv", sep = ''))
}


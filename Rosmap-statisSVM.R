library(mixOmics)
library(foreach)
library(kernlab)
library(pROC)
library(caret)
library(BiocManager)
library(MLmetrics)
library(mixKernel)
library(caret)
library(parallel)


# Import data -------------------------------------------------------------


mRNA_train =  read.csv("/data/ROSMAP/1_tr.csv", sep=",",header = FALSE)
meth_train =  read.csv("/data/ROSMAP/2_tr.csv", sep=",", header = FALSE)
miRNA_train =  read.csv("/data/ROSMAP/3_tr.csv", sep=",", header = FALSE)


mRNA_test =  read.csv("/data/ROSMAP/1_te.csv", sep=",", header = FALSE)
meth_test =  read.csv("/data/ROSMAP/2_te.csv", sep=",", header = FALSE)
miRNA_test =  read.csv("/data/ROSMAP/3_te.csv", sep=",", header = FALSE)

names_mRNA =  read.csv("/data/ROSMAP/1_featname.csv", sep=",", header = FALSE)
names_meth =  read.csv("/data/ROSMAP/2_featname.csv", sep=",", header = FALSE)
names_miRNA = read.csv("/data/ROSMAP/3_featname.csv", sep=",", header = FALSE)



label_train = read.csv("/data/ROSMAP/labels_tr.csv", sep=",",header = FALSE)
label_train_numeric = label_train$V1
label_train =  as.factor(label_train_numeric)


label_test = read.csv("/data/ROSMAP/labels_te.csv", sep=",", header = FALSE)
label_test_numeric = label_test$V1
label_test =  as.factor(label_test_numeric)

labels <- c(label_train_numeric,label_test_numeric)
# Convert mRNA_train to a data frame with column names
mRNA_train <- as.data.frame(mRNA_train)
names_mRNA <- as.vector(names_mRNA$V1)
colnames(mRNA_train) <- names_mRNA

meth_train  <- as.data.frame(meth_train )
names_meth <- as.vector(names_meth$V1)
colnames(meth_train ) <- names_meth

miRNA_train  <- as.data.frame(miRNA_train )
names_miRNA <- as.vector(names_miRNA$V1)
colnames(miRNA_train ) <- names_miRNA

mRNA_test <- as.data.frame(mRNA_test)
colnames(mRNA_test) <- names_mRNA

meth_test  <- as.data.frame(meth_test )
colnames(meth_test ) <- names_meth

miRNA_test  <- as.data.frame(miRNA_test)
colnames(miRNA_test ) <- names_miRNA

training_data <- list(mRNA = mRNA_train, meth = meth_train, miRNA = miRNA_train)
test_data <- list(mRNA = mRNA_test, meth = meth_test, miRNA = miRNA_test)

labels <- c(label_train_numeric,label_test_numeric)
ROSMAP_mRNA <- rbind(mRNA_train, mRNA_test)
ROSMAP_meth <- rbind(meth_train, meth_test)
ROSMAP_miRNA <- rbind(miRNA_train, miRNA_test)
DATASET_ROSMAP <- cbind(labels,ROSMAP_mRNA, ROSMAP_meth,ROSMAP_miRNA  )

DATASET_ROSMAP <- as.data.frame(DATASET_ROSMAP )
DATASET_ROSMAP_data <- DATASET_ROSMAP[,-1]
x <- as.matrix(DATASET_ROSMAP_data)
n <- nrow(x)
p <- ncol(x)
y <- DATASET_ROSMAP[,1]



# Parametrs tuning --------------------------------------------------------
#parameters
C_values <- c(1, 5, 10, 20,22, 25)
sigma_values <- c(0.0005,0.00005, 0.0001, 0.00001, 0.005)


sigma_combinations <- expand.grid(sigma1 = sigma_values, sigma2 = sigma_values, sigma3 = sigma_values)

# Define the number of cores for parallel execution
numCores <- 3


processIteration <- function(e) {
  cat("Run n:", e, '\n')
  cat("Starting Run:", e, "out of 5\n")
  set.seed(e - 1)
  ntrain <- 246      
  split_index <- createDataPartition(y, p = 0.7, list = FALSE)
  tindex <- split_index[1:ntrain]
  nfold <- 5
  
  # Initialize arrays for storing metrics
  Acc_fold <- array(0, dim = c(length(C_values), nrow(sigma_combinations)))
  Acc_fold <- as.data.frame(Acc_fold)
  
  dimnames(Acc_fold) <- list(C_values, seq_len(nrow(sigma_combinations)))
  # Store the results for 5 folds
  cv_results_ACC  <- list()
  
  # Loop for cross-validation folds
  for (f in 1:nfold) {
    cat("Fold n:", f, '\n')
    
    fold_indices <- createMultiFolds(y[tindex], k = nfold)
    fold_train <- tindex[fold_indices[[f]]]
    fold_val <- tindex[-fold_indices[[f]]]
    
    # Loop for different parameter combinations
    for (C in C_values) {
      for (comb in 1:nrow(sigma_combinations)) {
       
        sigma1 <- sigma_combinations$sigma1[comb]
        sigma2 <- sigma_combinations$sigma2[comb]
        sigma3 <- sigma_combinations$sigma3[comb]
        
        ROSMAP_mRNA_foldtr <- scale(ROSMAP_mRNA[fold_train,])
        X_mean_1 <- colMeans(ROSMAP_mRNA[fold_train,])
        X_std_1 <- apply(ROSMAP_mRNA[fold_train,], 2, sd)
        
        ROSMAP_meth_foldtr <- scale(ROSMAP_meth[fold_train,])
        X_mean_2 <- colMeans(ROSMAP_meth[fold_train,])
        X_std_2 <- apply(ROSMAP_meth[fold_train,], 2, sd)
        
        ROSMAP_miRNA_foldtr <- scale(ROSMAP_miRNA[fold_train,])
        X_mean_3 <- colMeans(ROSMAP_miRNA[fold_train,])
        X_std_3 <- apply(ROSMAP_miRNA[fold_train,], 2, sd)
        
        
        #kernel for fold train
        
        mRNA.kernel_ftrain <- compute.kernel(ROSMAP_mRNA_foldtr, kernel.func = "gaussian.radial.basis", sigma = sigma1, scale = FALSE)
        meth.kernel_ftrain <- compute.kernel(ROSMAP_meth_foldtr, kernel.func = "gaussian.radial.basis", sigma = sigma2, scale =  FALSE)
        miRNA.kernel_ftrain <- compute.kernel(ROSMAP_miRNA_foldtr, kernel.func = "gaussian.radial.basis", sigma = sigma3, scale =  FALSE)
        
        #Find the meta kernel and  the weights according to combine.kernels on the fold-train data
        Rosmap_fused_ftrain <- combine.kernels(mRNA = mRNA.kernel_ftrain, meth = meth.kernel_ftrain, miRNA = miRNA.kernel_ftrain, method = "STATIS-UMKL", knn = 10)
        weights_ftrain <- Rosmap_fused_ftrain$weights
        myK_ftrain <- weights_ftrain[1]* mRNA.kernel_ftrain$kernel + weights_ftrain[2]* meth.kernel_ftrain$kernel + weights_ftrain[3]*miRNA.kernel_ftrain$kernel
        
        
        #Fit an svm on the fold train data
        svp <- ksvm(myK_ftrain, y[fold_train], type = "C-svc", kernel = 'matrix', C = C)
        
        
        #kernel for the entire train scaled with values of fold train
        ROSMAP_mRNA_tr <- sweep(ROSMAP_mRNA[tindex,], 2, X_mean_1, "-")
        ROSMAP_mRNA_tr <- sweep(ROSMAP_mRNA_tr, 2, X_std_1, "/")
        
        ROSMAP_meth_tr <- sweep(ROSMAP_meth[tindex,], 2, X_mean_2, "-")
        ROSMAP_meth_tr <- sweep(ROSMAP_meth_tr, 2, X_std_2, "/")
        
        ROSMAP_miRNA_tr <- sweep(ROSMAP_miRNA[tindex,], 2, X_mean_3, "-")
        ROSMAP_miRNA_tr <- sweep(ROSMAP_miRNA_tr, 2, X_std_3, "/")
        
        
        mRNA.kernel <- compute.kernel(ROSMAP_mRNA_tr, kernel.func = "gaussian.radial.basis", sigma = sigma1, scale = FALSE)
        meth.kernel <- compute.kernel(ROSMAP_meth_tr, kernel.func = "gaussian.radial.basis", sigma = sigma2, scale = FALSE)
        miRNA.kernel <- compute.kernel(ROSMAP_miRNA_tr, kernel.func = "gaussian.radial.basis", sigma = sigma3, scale = FALSE)
        
        #obtain fused kernel for the entire  train based on weights optained on the fold train subset
        myK <- weights_ftrain[1]*mRNA.kernel$kernel + weights_ftrain[2]*meth.kernel$kernel + weights_ftrain[3]*miRNA.kernel$kernel
        
        relative_fold_train <- match(fold_train, tindex)
        relative_fold_val <- match(fold_val, tindex)
        
        # Use these relative indices to subset the kernel matrix for validation
        testK <- myK[relative_fold_val, relative_fold_train]
        testK <- testK[, SVindex(svp), drop = FALSE]
        ypred <- predict(svp, as.kernelMatrix(testK))
        ypredf <- as.factor(ypred)
        
        # response: vector of actual outcomes
        # predictor: vector of probabilities, one for each observation
        
        acc <- confusionMatrix(ypredf, as.factor(y[fold_val]))$overall['Accuracy']
        
        
        C_index <- match(C, C_values)
        Acc_fold[C_index, comb] <- acc
      }
    }
    # Store fold results
    cv_results_ACC[[f]] <- Acc_fold
    
  }
  
  num_rows <- nrow(cv_results_ACC[[f]])
  num_cols <- ncol(cv_results_ACC[[f]])
  mean_ACC <- as.data.frame(matrix(0, nrow = num_rows, ncol = num_cols))
  
  
  # Calculate the mean for each position in the dataframe
  for (i in 1:num_rows) {
    for (j in 1:num_cols) {
      # Calculate the mean for the corresponding positions in the datasets
      mean_value <- mean(sapply(cv_results_ACC, function(dataset) dataset[i, j]))
      mean_ACC[i, j] <- mean_value
    }
  }
  dimnames(mean_ACC) <- list(C_values, seq_len(nrow(sigma_combinations)))
  
  #find the best combination
  
  mean_ACC_matrix <- as.matrix(mean_ACC)
  best_combination <- which.max(mean_ACC_matrix)
  
  x <- which(mean_ACC_matrix == max(mean_ACC_matrix), arr.ind = TRUE)
  x
  best_C <- C_values[x[,1]]
  best_sigma_comb <- sigma_combinations[x[,2],]
  
  ##compute the kernel combination on the entire train set
  ROSMAP_mRNA_tr <- scale(ROSMAP_mRNA[tindex,])
  X_mean_1 <- colMeans(ROSMAP_mRNA[tindex,])
  X_std_1 <- apply(ROSMAP_mRNA[tindex,], 2, sd)
  
  ROSMAP_meth_tr <- scale(ROSMAP_meth[tindex,])
  X_mean_2 <- colMeans(ROSMAP_meth[tindex,])
  X_std_2 <- apply(ROSMAP_meth[tindex,], 2, sd)
  
  ROSMAP_miRNA_tr <- scale(ROSMAP_miRNA[tindex,])
  X_mean_3 <- colMeans(ROSMAP_miRNA[tindex,])
  X_std_3 <- apply(ROSMAP_miRNA[tindex,], 2, sd)
  
  mRNA.kernel_t <- compute.kernel(ROSMAP_mRNA_tr, kernel.func = "gaussian.radial.basis", sigma = best_sigma_comb$sigma1[1], scale = FALSE)
  meth.kernel_t <- compute.kernel(ROSMAP_meth_tr, kernel.func = "gaussian.radial.basis", sigma = best_sigma_comb$sigma2[1], scale =  FALSE)
  miRNA.kernel_t <- compute.kernel(ROSMAP_miRNA_tr, kernel.func = "gaussian.radial.basis", sigma = best_sigma_comb$sigma3[1], scale = FALSE)
  
  Rosmap_fused_t <- combine.kernels(mRNA = mRNA.kernel_t, meth = meth.kernel_t, miRNA = miRNA.kernel_t, method = "STATIS-UMKL", knn = 10)
  
  #Store the weights
  weights_t <- Rosmap_fused_t$weights
  myK_t <- weights_t[1]*mRNA.kernel_t$kernel +  weights_t[2]*meth.kernel_t$kernel +  weights_t[3]*miRNA.kernel_t$kernel
  
  #fit svm on the train set
  svp <- ksvm(myK_t, y[tindex], type = "C-svc", kernel = 'matrix', C = best_C[1],prob.model = TRUE)
  
  ##kernel for the entire DATASET scaled with values of train
  ROSMAP_mRNA_t <- sweep(ROSMAP_mRNA, 2, X_mean_1, "-")
  ROSMAP_mRNA_t <- sweep(ROSMAP_mRNA_t, 2, X_std_1, "/")
  
  ROSMAP_meth_t <- sweep(ROSMAP_meth, 2, X_mean_2, "-")
  ROSMAP_meth_t <- sweep(ROSMAP_meth_t, 2, X_std_2, "/")
  
  ROSMAP_miRNA_t <- sweep(ROSMAP_miRNA, 2, X_mean_3, "-")
  ROSMAP_miRNA_t <- sweep(ROSMAP_miRNA_t, 2, X_std_3, "/")
  
  
  mRNA.kernel <- compute.kernel(ROSMAP_mRNA_t, kernel.func = "gaussian.radial.basis", sigma =  best_sigma_comb$sigma1[1], scale =  FALSE)
  meth.kernel <- compute.kernel(ROSMAP_meth_t, kernel.func = "gaussian.radial.basis", sigma =  best_sigma_comb$sigma2[1], scale =  FALSE)
  miRNA.kernel <- compute.kernel(ROSMAP_miRNA_t, kernel.func = "gaussian.radial.basis", sigma =  best_sigma_comb$sigma3[1], scale =  FALSE)
  
  
  
  test_indices <- setdiff(1:n, tindex)  # Indices of test samples
  
  testK_mRNA.kernel<- mRNA.kernel$kernel[test_indices, tindex]
  testK_meth.kernel<- meth.kernel$kernel[test_indices, tindex]
  testK_miRNA.kernel<- miRNA.kernel$kernel[test_indices, tindex]
  
  
  
  myK <-  weights_t[1]*testK_mRNA.kernel +  weights_t[2]*testK_meth.kernel +  weights_t[3]*testK_miRNA.kernel
  
  
  # Predict on the test set
  
  testK <- myK[, SVindex(svp), drop = FALSE]
  ypred <- predict(svp, as.kernelMatrix(testK))
  yprob <- predict(svp, as.kernelMatrix(testK), type="probabilities")
  positive_class_probabilities <- yprob[, 2]
  # Check the performance on the test set
  y_factor <- as.factor(ypred)
  acc <- confusionMatrix(y_factor, as.factor(y[test_indices]))$overall['Accuracy']
  F1 <- F1_Score(y[test_indices], ypred)
  
  roc_c <- roc(y[test_indices], positive_class_probabilities)
  AUC <- auc(roc_c )
  cat("Completed Run:", e, "out of 5\n")
  
  # Return the results of this iteration
  return(list(Accuracy_test = acc, Auc = AUC, F1 =  F1))
}
test_result <- processIteration(1)
print(test_result)
# Run the iterations in parallel
results <- mclapply(1:5, processIteration, mc.cores = numCores)

# Extract and combine results
Accuracy_test <- sapply(results, function(x) x$Accuracy_test)
F1 <- sapply(results, function(x) x$F1)
Auc <- sapply(results, function(x) x$Auc)

# Output the results
print(Accuracy_test)
print(F1)
print(Auc)

# Calculate and output the mean and standard deviation
MeanAccTest <- mean(Accuracy_test)
MeanF1 <- mean(F1)
MeanAuc <- mean(Auc)
sdAccTest <- sd(Accuracy_test)
sdF1 <- sd(F1)
sdAuc <- sd(Auc)
cat('Mean Accuracy:', MeanAccTest, 'Sd Accuracy:', sdAccTest, 'Mean F1:', MeanF1, 'Sd F1:', sdF1,
    'Mean Auc:', MeanAuc, 'Sd Auc:', sdAuc,'\n')

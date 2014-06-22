Practical Machine Learning
========================================================
## Prediction Assignment Writeup
### Introduction
The goal of this project is to predict human activity based on wearable accelerometers' data. The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har].

### Data cleaning
We will focus on prediction test data, therefore  we find all fields in the test dataset, that contains no data or data that cannot be used (like row number) and remove those fields from train dataset. This report created only for code demonstration therefore we used only part of train data (not all dataset). 

```r
# load data
t <- read.csv("..//pml-testing.csv", na.strings = c("NA", ""))  # test dataset
d <- read.csv("..//pml-training.csv", na.strings = c("NA", ""))  # train dataset
sprintf("Number of variables in train dataset: %i", ncol(d) - 1)
```

```
## [1] "Number of variables in train dataset: 159"
```

```r
# sample train dataset to reduce calculation time
d.sample <- sample(nrow(d), 300)
d <- d[d.sample, ]
# Find fields in test dataset that have no data
t.NA <- apply(t, 2, function(x) {
    sum(is.na(x))
})
t.NA <- which(t.NA < nrow(t))
# Remove fields that have no data from test and train dataset
t <- t[, t.NA]
d <- d[, t.NA]
t <- t[, -1]
d <- d[, -1]
sprintf("Number of selected predictors: %i", ncol(d) - 1)
```

```
## [1] "Number of selected predictors: 58"
```

### Preprocess data
We used PCA to reduce number of predictors. PCA required all predictors to be numeric type, therefore non-numeric variables was converted to a numeric type first.   

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
# Convert non-numeric variable
for (i in c(1, 4, 5)) {
    d[, i] <- as.numeric(d[, i])
    t[, i] <- as.numeric(t[, i])
}
d.preProc <- preProcess(d[, -59], method = "pca", thresh = 0.9)
training <- predict(d.preProc, d[, -59])
testing <- predict(d.preProc, t[, -59])
d.preProc
```

```
## 
## Call:
## preProcess.default(x = d[, -59], method = "pca", thresh = 0.9)
## 
## Created from 300 samples and 58 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 21 components to capture 90 percent of the variance
```

### Training 
The random forest method ('rf') selected for prediction. Train data splited to have data for cross-validation.

```r
inTrain <- createDataPartition(d[, 59], p = 3/4)[[1]]
modelFit <- train(d[inTrain, 59] ~ ., data = training[inTrain, ], method = "rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modelFit
```

```
## Random Forest 
## 
## 228 samples
##  20 predictors
##   5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 228, 228, 228, 228, 228, 228, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.5       0.4    0.05         0.06    
##   10    0.5       0.4    0.05         0.07    
##   20    0.5       0.4    0.06         0.08    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```


### Cross-validation
Part of data that not used in train used for closs-validation.

```r
confusionMatrix(d[-inTrain, 59], predict(modelFit, training[-inTrain, ]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  A  B  C  D  E
##          A 15  1  2  2  0
##          B  4  4  3  1  1
##          C  4  1  6  0  1
##          D  3  0  1  7  2
##          E  2  0  2  3  7
## 
## Overall Statistics
##                                       
##                Accuracy : 0.542       
##                  95% CI : (0.42, 0.66)
##     No Information Rate : 0.389       
##     P-Value [Acc > NIR] : 0.00609     
##                                       
##                   Kappa : 0.414       
##  Mcnemar's Test P-Value : 0.51323     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.536   0.6667   0.4286   0.5385   0.6364
## Specificity             0.886   0.8636   0.8966   0.8983   0.8852
## Pos Pred Value          0.750   0.3077   0.5000   0.5385   0.5000
## Neg Pred Value          0.750   0.9661   0.8667   0.8983   0.9310
## Prevalence              0.389   0.0833   0.1944   0.1806   0.1528
## Detection Rate          0.208   0.0556   0.0833   0.0972   0.0972
## Detection Prevalence    0.278   0.1806   0.1667   0.1806   0.1944
## Balanced Accuracy       0.711   0.7652   0.6626   0.7184   0.7608
```


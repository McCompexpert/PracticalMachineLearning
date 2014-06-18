Using a modern machine learning method for prediction of human activity.
========================================================
# Synopsis
We are given a big dataset containing a big number of attributes discribing different measurements from accelometers on the belt, forearm, arm, and dumbell of six participants. They are asked to perform excercises in correct and incorrect manner. We split the training dataset in two parts for training and testing. We are using Random Forest to train our prediction model with cross-validation. We applied the model on the final submission dataset of 20 cases. All 20 cases have been predicted correctly - herewith, we achieved the out of sample accuracy of 100%.

# Accessing Data from the Web
We are using direct web links to access the data, download it, store locally and , finaly, cache the variables in .Rdata file to speed up our analysis by repeated data loads. We are raplacing NA strings, empty strings "" and #DIV/0!" with proper NAs in read.csv function.


```r
# Use tempfile variable for storing downloaded data locally
temp1 <- tempfile()
temp2 <- tempfile()
# download the training dataset by using the web link
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
    temp1)
# read in the training data
raw_training <- read.csv(temp1, header = TRUE, sep = ",", dec = ".", na.strings = c("NA", 
    "", "#DIV/0!"))

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
    temp2)

raw_testing <- read.csv(temp2, header = TRUE, sep = ",", , dec = ".", na.strings = c("NA", 
    "", "#DIV/0!"))

# cache the downloadet data locally
save(raw_training, file = "raw_training.RData")
save(raw_testing, file = "raw_testing.RData")
```

# Data pre-processing
We put aside the testing data set for our final model validation and will work with training dataset only. Our dataset contains 19.622 records with 160 different measurements (attributes), many of them contain NA values. 

```r
setwd("~/GitHub/PracticalMachineLearning")
# load of previously chached data
load("raw_training.RData")
load("raw_testing.RData")
# Summary of the training dataset
dim(raw_training)
```

```
## [1] 19622   160
```

First, we get rid of all columns with at least one NA value. Second, we get rid of the first seven columns (user name, timestamps, window descriptors) as they all are not measurement results.Our original dataset is reduced to 53 variables. This dataset we will use in our analysis.


```r
# reduce the training data set by deleting all columns containing NAs
training.reduced <- raw_training[sapply(raw_training, function(x) !any(is.na(x)))]
# reduce the training set by non-measurement variables
training.reduced <- training.reduced[, c(-1, -2, -3, -4, -5, -6, -7)]
# We reduced
dim(raw_training)
```

```
## [1] 19622   160
```

```r
# construct our training dataset
training <- training.reduced

# Prepare the submission dataset by selecting the same column names as from
# training
SELECT <- colnames(training[, c(-53)])
# reduce testing set to the training columns
raw_testing.reduced <- raw_testing[, c(-160)]
# This submission dataset will be used for the final predicion
raw_testing.reduced <- raw_testing.reduced[, SELECT]
```

# Model building procedure
We are using the caret package for our data analysis. First, we are splitting our trainin dataset in two parts with 80%/20% rule. We will use 80% for our model training and the rest 20% for model validation. Finally, the validated model will be applied on submission dataset to predict 20 activity classes.


```r
# library load
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
# create a 80%/20% subtrain and subtest
trainIndex = createDataPartition(training$classe, p = 0.8, list = FALSE)
training = training[trainIndex, ]
testing = training[-trainIndex, ]
```

Next, we contruct the trainControl function using cross-validation parameter and allowing for using our dual-core processor to seep up calculation. We use Random Forest method with 4-fold cross-validation for our model training. As the calculation lasts about 30 minutes on our Win7, 32bit, 2Gb RAM notebook, we store the model estimate locally in the variable g1.


```r
# define a control function using the 4-fold cross-validation method
ctrl <- trainControl(method = "cv", number = 4, allowParallel = TRUE)
# train the model
g1 <- train(classe ~ ., data = training, method = "rf", trControl = ctrl)
save(g1, file = "g1.RData")
```

Next, we using the function predict on the validation dataset (20%) and calculate the prediction accuracy. We achieved an remarkable accuracy as of 0.99 on our validation dataset, erring in only very low number of cases (see confusion matrix). This is good enough to make a conclusion that our model can be applied on the final submission dataset to predict 20 activities.


```r
# load chached model estimate from Random Forest run on the traing dataset
# containing 15699 records (80%).
load("g1.RData")
# apply the RF model on validation dataset (20%, 3141 records)
testPred <- predict(g1, testing)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# calculate the prediction accuracy
postResample(testPred, testing$classe)
```

```
## Accuracy    Kappa 
##   0.9997   0.9996
```

```r
# Confussion Matrix shows that our model is erring in only one case.
confusionMatrix(testPred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 900   1   0   0   0
##          B   0 577   0   0   0
##          C   0   0 566   0   0
##          D   0   0   0 520   0
##          E   0   0   0   0 580
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.998, 1)
##     No Information Rate : 0.286     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.998     1.00    1.000    1.000
## Specificity             1.000    1.000     1.00    1.000    1.000
## Pos Pred Value          0.999    1.000     1.00    1.000    1.000
## Neg Pred Value          1.000    1.000     1.00    1.000    1.000
## Prevalence              0.286    0.184     0.18    0.165    0.184
## Detection Rate          0.286    0.184     0.18    0.165    0.184
## Detection Prevalence    0.287    0.184     0.18    0.165    0.184
## Balanced Accuracy       1.000    0.999     1.00    1.000    1.000
```

# Results
## Out of sample error
Now, we are applying the Random Forest model, previously trained on 15K+ records, validated on validation dataset (3K+ records) and finaly applied on the submission dataset. All 20 cases have been classified correctly which lets us conclude that on this dataset we achieved an out of sample error as of zero. Whether this model will bring the same zero errors at classifying other new records, needs to be analyzed separately.


```r

finalPred <- predict(g1, raw_testing.reduced)
finalPred
# All 20 submission cases have been predictied correctly.
```



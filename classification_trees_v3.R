library(data.table)
library(ggplot2)
library(randomForest)
library(anytime) #Load library for date conversion
library(reshape2)
library(lattice)
library(ggvis)
library(rpart)
library(rpart.plot)
library(C50)
library(corrplot)
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(caret)
library(rpart)
library(Metrics)
library(caret)


ios_act_raw <- data.table(read.csv("ios_act.csv")) #Read data from file into a data table`

#Basic exploration
str(ios_act_raw)
dim(ios_act_raw) #The number of rows and columns
colnames(ios_act_raw) #The column names
summary(ios_act_raw) #Some descriptive statistics about the dataset
attributes(ios_act_raw) #The column names and the row names (Rownames are numerical in this example)
duplicated(ios_act_raw) #Check for duplicate names
sum(is.na(ios_act_raw)) #Check for missing values
ios_act_raw[, colSums(ios_act_raw != 0)>0] #Check for columns where all values like zero
prop.table(table(ios_act_raw$activity)) #Percentage distribution of activities

#Visual inspection
table(ios_act_raw$activity)# View the amount of data for each activity
barplot(table(ios_act_raw$activity),col= c("green","red","black","blue", "yellow"), names.arg = c("Unknown","Stationary","Walking","Running","Automotive")) #Plot the table
grid() #Add grid to the plot


#------------------------------------------------------------------------------------
#Data Preprocessing
#Data Pre-processing
#------------------------------------------------------------------------------------
#Remove columns from raw data dataset
ios_act<-ios_act_raw[,-c(1:4,12,13,17,21,25:47,49, 50:54)] #Remove columns with the sum of zero,time,std = zero
ios_act$activity<-factor(ios_act$activity)
str(ios_act)

#the columns are collapsed into a key and value relationship
ios_act.mlt <- data.table::melt(ios_act, id.vars="activity", measure.vars = c(1:16)) #Transform the columns into key and values
densityplot(~value|variable,data = ios_act.mlt, scales = list(x = list(relation = "free"), y = list(relation = "free")),adjust = 1.25, pch = "|", xlab = "Predictor")

#One and one predictor
density(ios_act$locationLatitude)
plot(density(ios_act$locationLatitude))
density(ios_act$gyroRotationZ)
plot(density(ios_act$gyroRotationZ))

#Columns with predictors
res=cor(ios_act.pred) #Compute a correlation matrix
corrplot(res, type="upper", order="hclust", tl.col = "black", tl.srt=45)

#Visual inspection
#pairs(ios_act.pred) #Look at a pairwise scatterplot

ios_act %>% ggvis(~locationLatitude, ~locationLongitude, fill = ~activity) %>% layer_points()


#Randomize the observations
ios_act.pred<-ios_act[,-c(17)] #A dataset with only the predictors
smp_size<-floor(0.8*nrow(ios_act)) #Sett sample size to 80% of the observations
set.seed(123)
ios_act_ind<-sample(seq_len(nrow(ios_act)),size=smp_size) #Create a ranmoized dataset
ios_act.train<-ios_act[ios_act_ind,] #Training dataset (80%)
ios_act.test<-ios_act[-ios_act_ind,] #Test dataset (20%)
ios_act.train.pred<-ios_act.train[,-c(17)] #A training dataset with only the predictors
ios_act.train.target<-ios_act.train[,c(17)] #A training dataset with only the target variable
ios_act.test.pred<-ios_act.test[,-c(17)] #A test dataset with only the predictors
ios_act.test.target<-ios_act.test[,c(17)] #A test dataset with only the target variable


"TBD"
#Data transformation of predictors
#PCA visualize the variation present in a dataset. you take a dataset with many variables, and you simplify 
#that dataset by turning your original variables into a smaller number of "Principal Components
ios_act.pca<-prcomp(ios_act.pred, center=TRUE, scale. = TRUE)
newpca.train<-predict(ios_act.pca, ios_act.train)
newpca.train<-data.frame(newpca.train, ios_act.train$activity)
newpca.test<-predict(ios_act.pca, ios_act.test)
newpca.test<-data.frame(newpca.train, ios_act.test$activity)

ios_act.train.pca <- prcomp(ios_act.train.pred, center=TRUE, scale. = TRUE)
ios_act.test.pca <- prcomp(ios_act.test.pred, center=TRUE, scale. = TRUE)


#Classification tree
m2<-C5.0(ios_act.train.pred, ios_act.train.target$activity)
summary(m2) #Review the output of the model
mp2<-predict(m2, ios_act.test) #Run the prediction
table(ios_act.test$activity, mp2) #Create a confusion matrix
#rmse(ios_act.test$activity, mp2) #Can be used to asess the various models. The lower value the better

#----------------Training-------------------------
#preprPredictors = preProcess(ios_act.train.pred, method=c("BoxCox", "center", "scale")) #Preprosess the original predictors
preprPredictors = preProcess(ios_act.train.pred, method=c("BoxCox", "center", "scale")) #Preprosess the original predictors
transformedPredTrain <- predict(preprPredictors, ios_act.train.pred)
pcacomp = preProcess(ios_act.train.pred, method=c("pca"))  #Preprosess predictors and generate PCA
transformedPredPCATrain <- predict(pcacomp, ios_act.train.pred) 
par(mfrow=c(1,2)); hist(modpred$locationLongitude); qqnorm(modpred$locationLongitude)
par(mfrow=c(1,2)); hist(modpredPCA$PC2); qqnorm(modpredPCA$PC2)

transTrainingDataset<-cbind(transformedPredTrain, ios_act.train.target)
#Review dataset after transformations
ios_act.mlt <- data.table::melt(transTrainingDataset, id.vars="activity", measure.vars = c(1:16)) #Transform the columns into key and values
densityplot(~value|variable,data = ios_act.mlt, scales = list(x = list(relation = "free"), y = list(relation = "free")),adjust = 1.25, pch = "|", xlab = "Predictor")


#----------------Test-----------------------------
#preprPredictorsTst = preProcess(ios_act.test.pred, method=c("BoxCox", "center", "scale")) #Preprosess the original predictors
preprPredictorsTst = preProcess(ios_act.test.pred, method=c("center", "scale")) #Preprosess the original predictors
transformedPredTest<-predict(preprPredictorsTst, ios_act.test.pred)
pcacompTst = preProcess(ios_act.test.pred, method=c("BoxCox", "center", "scale", "pca"))  #Preprosess predictors and generate
transformedPredPCATest<-predict(pcacompTst, ios_act.test.pred) 

ios_act.pca<-prcomp(ios_act.train.pred, center=TRUE, scale. = TRUE)



#Classification tree after preprosessing with Box Cox, Center and Scaling
#ios_act.train.predpca <- ios_act.train.pca$x #Keep 
#predict(ios_act.pca, data.table(ios_act.test.target))
m2.1<-C5.0(transformedPredTrain, ios_act.train.target$activity)
summary(m2.1) #Review the output of the model
modTestData<-cbind(transformedPredTest, ios_act.test.target)
mp2.1<-predict(m2.1, modTestData) #Run the prediction. Y skal være preprosessert datasett. ModpredTst ,angler activity
table(mp2.1, modTestData$activity) #Create a confusion matrix

m2.2<-C5.0(transformedPredPCATrain, ios_act.train.target$activity)
summary(m2.2) #Review the output of the model
modTestDataPCA<-cbind(transformedPredTest, ios_act.test.target)
mp2.2<-predict(m2.2, modTestDataPCA) #Run the prediction. Y skal være preprosessert datasett. ModpredTst ,angler activity
table(mp2.2, modTestDataPCA$activity) #Create a confusion matrix

#Run C5.0 rule based
m2.3<-C5.0(ios_act.train.pred, ios_act.train.target$activity, rules = TRUE)
summary(m2.3) #Review the output of the model
mp2.3<-predict(m2.3, ios_act.test) #Run the prediction
table(ios_act.test$activity, mp2.3) #Create a confusion matrix

#Classification tree cost
cost_mat <- matrix(0,0,0,0,1,2,0,2,2,2,3,3,3,0,3,4,4,4,0,4,5,5,5,0,5), nrow = 5)
rownames(cost_mat) <- colnames(cost_mat) <- c("Unknown","Stationary","Walking","Running","Automotive")
cost_mat


m2.4<-C5.0(ios_act.train.pred, ios_act.train.target$activity, costs = cost_mat)
summary(m2) #Review the output of the model
mp2.4<-predict(m2.4, ios_act.test) #Run the prediction
table(ios_act.test$activity, mp2) #Create a confusion matrix



#Random Forrest (Standard values mtry=4 and number of trees = 500)
m3<-randomForest(ios_act.train.pred, ios_act.train.target$activity, importance = TRUE, na.action = na.omit)
m3$confusion
summary(m3)
print(m3)
attributes(m3) #Show the attributes of the model
mp3<-predict(m3, ios_act.test)
attributes(mp3)
table(ios_act.test$activity, mp3)

m3.1<-randomForest(transformedPredTrain, as.factor(ios_act.train.target$activity), importance = TRUE, na.action = na.omit)
mp3.1$confusion #Confusing matrix for the modelsummary(m3.1)
print(m3.1)
attributes(m3.1) #Show the attributes of the model
mp3.1<-predict(m3.1, ios_act.test.pca)
table(ios_act.test$activity, mp3.1)




confusionMatrix(mp3, is.factor(ios_act.test$activity))
summary(mp3)  
rmse(ios_act.test$activity, mp3) #Can be used to asess the various models. The lower value the better


plot(ios_act.pca$x[,1], ios_act.pca$x[,2]) #Plot PC1 and PC2
pca.var<-ios_act.pca$sdev^2
pca.var.per<-round(pca.var/sum(pca.var)*100,1)
barplot(pca.var.per, main = "Plot", xlab="Principal component", ylab="Percentage Variation") #Percentage of variation for each PC



#XGBoost

#Parameters
xgb_params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = nc)
watchlist <- list(trein)

m4 <- xgb.train(params=xgb_params, data=train_matrix, nrounds = 100, watchlist =)
m4

#lage nytt datasett
#ios_act$loggingTime<-anytime(ios_act$loggingTime)
#rowsum(ios_act,ios_act$activitio
#apply(anytime(ios_act$loggingTime))
#ios_act$loggingTimeios



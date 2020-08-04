# ML-R-Project
My signature project for DA5030 Data Mining/Machine Learning in R Studio
I am predicting the popularity and sales of video games
The data set is taken from kaggle: https://www.kaggle.com/gregorut/videogamesales


---
title: "Signature Project Varun Sriram DA5030"
output:
  html_document:
    df_print: paged
---


Please Install these Libraries before execution
```{r}
#install.packages("GGally")
#install.packages("DMwR")
#install.packages("psych")
#install.packages("caret")
#install.packages("class")
#install.packages("gmodels")
#install.packages("ROCR")
#install.packages("AUC")
#install.packages("gmodels")
#install.packages("RWeka")
#install.packages("caretEnsemble")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("e1071")




library(GGally)
library(DMwR)
library(psych)
library(caret)
library(class)
library(gmodels)
library(ROCR)
library(AUC)
library(gmodels)
library(RWeka)
library(caretEnsemble)
library(rpart)
library(rpart.plot)
library(e1071)

```






```{r}
sales<-read.csv("vgsales.csv") #reading the data set with na strings. 

#checking certain columns for N/A values
str(sales)
summary(sales$Year)
which(is.na(sales$Genre))
which(is.na(sales$Publisher))

#As we can see we have N/A values as string, we will import the data set again

sales<-read.csv("vgsales.csv", na.strings = c("N/A"), stringsAsFactors = FALSE) #reading the data set with "N/A" as" strings. 

#checking certain columns for N/A values
str(sales)
summary(sales$Year)
summary(as.factor(sales$Genre))
summary(as.factor(sales$Publisher))
```
As you can see, The data set has:
- 11 columns
- 16598 observations
- Median for Year is 2007 and mean is 2006 not necessary
- Most popular genre
- Publisher with most games released
- Widespread used platform





Now we move to data Imputation
- As we can see that we have N/A as a string in the columns Year and Publisher
- Using mode to replace N/A vlues in year, since it is the best option as we can consider that thre was one point in time where games were popular and a lot of games were released
- Considering N/A values in Publisher as a separate Publisher by itself. Since it is impossible to predict which publihser has done what, I have taken this decision. In the gaming industry, games without publishers are called "Indie Game Studios" Thus we will consider N/A as a group of indie game studios
```{r}

summary(sales$Year)

#finding mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#replacing year N/A's using mode value
modeforsales  <- Mode(as.numeric(sales$Year)) 
sales$Year[is.na(sales$Year)] <- modeforsales
summary(sales$Year)

#replacing publisher N/A with indie
sales$Publisher[is.na(sales$Publisher)] <- 'Indie'

sales$Publisher <- as.factor(sales$Publisher)
sales$Genre <- as.factor(sales$Genre)
sales$Platform <- as.factor(sales$Platform)

#removing row number and game names
sales <- sales[-1]
sales <- sales[-1]

```
Data Plots
```{r}
barplot(table(sales$Genre), main = "Games Genre nos") #bar plot for game genre
barplot(table(sales$Platform), main = "Games Platform nos") #barplot for platform
hist(sales$Year) #a histogram for years and getting the sales

#lets check the distribution of each region sales with respec to global sales 
with(sales ,plot(NA_Sales,Global_Sales))
with(sales ,plot(EU_Sales,Global_Sales))
with(sales ,plot(JP_Sales,Global_Sales))
with(sales ,plot(Other_Sales,Global_Sales))

#A global sales vs year plot
plot(sales$Global_Sales, sales$Year, main="Genre vs Year", sub="subtitle", xlab="Genre", ylab="Year")
```
Here we will look at some data plots, in the last plot, genre vs year, we can see that majority of the games have been released in the 2000-2010 period. Which seems legit because that was the phase where there were many new companies developing because of the American recession. It had led to various new industries popping up! Some of them were new independent companies, in some situations, companies merged to form new ones (medium sized). Bigger companies bought nsmaller companies, extended their strength and invested on many newer games. 




Check for outliers. 
But I feel that in this data set which is about sales, the values for global sales are true and thus, I wont remove the outliers.
We will still check for outliers in global sales
```{r}

hist(as.numeric(sales$Global_Sales))
sd <- sd(sales$Global_Sales)
mean <- mean(sales$Global_Sales)
z <- mean-sales$Global_Sales
z<- z/sd
mean(z)
z<-abs(z)

outliers <- z[which(z > 3)]
outliers<- which(z > 3)
outliers


#visualize outlier
boxplot(sales$Global_Sales, main="With outliers")

```



Now we will derive two features:
- A playertype, after extensive online searching, I have come to the conclusion that some game genres have most games as single player. Now we will assign "Single" or "Multi" to denote the player type depending on the genre of the game
- Along with that we will assign degree of popularity depending on the global sales value of each game ranging from not, less, mudieum and very. This will be used in the models which will predict categories only. Like Decision trees etc

```{r}
#feature engineering: deriving a feature
sales$PlayerType <- 0
sales$Popularity <- 0


xx<- c("Action", "Adventure", "Misc", "Platform", "Puzzle", "Simualtion")#vector to store genre that will be checked to set playertype
a <- ifelse(sales$Global_Sales < 5, sales$Popularity <- "Less", sales$Popularity <- "Very") #popularity

c <- ifelse(sales$Genre == xx , sales$PlayerType <- "Single", sales$PlayerType <- "Multi") #playertype

sales$Popularity <- as.factor(a)
sales$PlayerType <- as.factor(c)

```



We will not be using dummy codes in our models, but to display that I have learn to create dummy codes, I will be creating new data frame which will contain the new dummy codes for column "Genre".

```{r}
#dummy code for genre 

dummycodes <-  dummy.code(sales$Genre)
sales_dummycodes <- data.frame(sales,dummycodes)
sales_dummycodes

```

Now we will be normalizing the data sets using min max method and store is in a new data frame
```{r}

#check for normalization, ask professor, just for the concept of normalization
normalize <- function(x) {
return ((x - min(x)) / (max(x) - min(x))) } #normalizing the function with Min-Max method


sales_normalized <- sales


sales_normalized[5:9] <- as.data.frame(lapply(sales[5:9], normalize))
  
```



Principle Component Analysis
```{r}
#PCA 
scaledsales <- scale(sales[,5:9], center = TRUE, scale = TRUE)
q<-cbind(scaledsales, sales$Global_Sales)
pca<-princomp(q,scores = TRUE,cor=TRUE)
loadings(pca)
plot(pca)
biplot(pca)

```
From the PCA, we can see that:
- NA sales and EU sales are closely important for contribution towards Global Sales
- Other Sales are next after NA and EU Sales
- Japan is the lowest in this situation



Co-relation and Co-Linearity
```{r}
#for co-relation / colinearity 2
ggpairs(as.data.frame(scaledsales)) #numeric features


categorical_corelations<-cbind(sales$Platform, sales)
pairs.panels(categorical_corelations) #correlation with categorical features


```
From the above, we can confirm that NA_Sales affects the global sales the most. Then the next in line in EU, then other and then Japan sales.
But the Japan sales and Other sales correlation is not so wide. We can confirm that JP sales affects global sales as all the other states combined.


Splitting data sets into training and validation data

```{r}

#creating training and test data sets
#Without normalization

set.seed(250)
index <- createDataPartition(sales$Global_Sales, p=0.75, list = FALSE)  
sales_train<-sales[index,]
sales_valid<-sales[-index,]


#normalized values
set.seed(251)
index1<-createDataPartition(sales_normalized$Global_Sales,p=0.75,list=FALSE)
salesnormalized_train<-sales_normalized[index1,]
salesnormalized_valid<-sales_normalized[-index1,]


```


Knn Model
I have saved the model already for faster training purposes. You just need to run the readRDS() function to read the model saved. This will help the TA and prof to read models faster and not have to train it all over again!
You have to execute ONLY the prediction model and confusion matrix to find the accuracy and kappa for each model
```{r}
#kNN








#Knn on normalized data set
set.seed(400)
training_control_normalized <- trainControl(method = "repeatedcv", number=10, repeats = 3)
knn_salesnormalized <- readRDS("./knn_salesnormalized_model.rds")
knn_salesnormalized <- train(Popularity ~., data = salesnormalized_train, method = "knn", metric = "Kappa", trControl = training_control_normalized)

knn_pred_salesnormalized <- predict(knn_salesnormalized, newdata = salesnormalized_valid)

confusionMatrix(knn_pred_salesnormalized,salesnormalized_valid$Popularity)
saveRDS(knn_salesnormalized, "./knn_salesnormalized_model.rds")



#knn on non-normalized data set
set.seed(400)
training_control <- trainControl(method = "repeatedcv", number=10, repeats = 3)
knn_sales<- readRDS("./knn_sales.rds")
knn_sales <- train(Popularity ~. , data = sales_train, method = "knn",metric = "Kappa", trControl = training_control)

knn_pred_sales <- predict(knn_sales, newdata = sales_valid)
confusionMatrix(knn_pred_sales, sales_valid$Popularity)
saveRDS(knn_sales, "./knn_sales.rds")



#kfold tuning on normalized data set
set.seed(400)
training_control_normalized_tuned <- trainControl(method = "repeatedcv", number=10, repeats = 3)
knn_salesnormalized_tuned<- readRDS("./knn_salesn_t.rds")
knn_salesnormalized_tuned <- train(Popularity ~.,data=salesnormalized_train,method="knn",metric = "Kappa", trControl=training_control_normalized_tuned, tuneLength = 5 )


knn_pred_salesnormalized_tuned <- predict(knn_salesnormalized_tuned, newdata = salesnormalized_valid)
summary(knn_pred_salesnormalized_tuned)

confusionMatrix(knn_pred_salesnormalized_tuned,salesnormalized_valid$Popularity)
saveRDS(knn_salesnormalized_tuned, "./knn_salesn_t.rds")


#tuning by holdout method on normalized data set
set.seed(400)
training_control_holdout <- trainControl(method = "LGOCV", number=10)
knn_sales_normalized_holdout<- readRDS("./knn_sales_n_h.rds")
knn_sales_normalized_holdout <- train(Popularity ~. , data = salesnormalized_train, method = "knn",metric = "Kappa", trControl = training_control_holdout)
knn_sales_normalized_holdout

knn_pred_salesnormalized_holdout <- predict(knn_sales_normalized_holdout, newdata = salesnormalized_valid)

confusionMatrix(knn_pred_salesnormalized_holdout,salesnormalized_valid$Popularity)
saveRDS(knn_sales_normalized_holdout, "./knn_sales_n_h.rds")
```
Simple KNN on normalized sales values: Accuracy = 0.9904 , Kappa = 0.1652
Simple KNN on original data set values : Accuracy = 0.9988, Kappa = 0.944
Knn with k-fold cross validation : Accuracy = 0.9904, Kappa = 0.1652
Knn with holdout method evaluation : Accuracy = 0.9904, Kappa = 0.1652 

SVM MODEL
```{r}
set.seed(102)
svm_model_onnorm<- svm(Popularity~., data = salesnormalized_train, scale=FALSE, kernel = "linear")

svm_predict_onnorm<-predict(svm_model_onnorm,salesnormalized_valid)
confusionMatrix(svm_predict_onnorm,salesnormalized_valid$Popularity)

set.seed(102)
svm_model<- svm(Popularity~., data = sales_train, scale=FALSE, kernel = "linear")

svm_predict<-predict(svm_model,sales_valid)
confusionMatrix(svm_predict,sales_valid$Popularity)

set.seed(102)
svm_model_kfold3<- svm(Popularity~., data = salesnormalized_train, scale=FALSE, kernel = "linear", cross = 3)


svm_predict_kfold_norm3<-predict(svm_model_kfold3,salesnormalized_valid)
confusionMatrix(svm_predict_kfold_norm3,salesnormalized_valid$Popularity)

```
From SVM models:
SVM on normalized data: Accuracy= 0.9894 , Kappa 0
SVM on non nomalized data: Accuracy=1 , Kappa = 1
SVM k -fold closs validation (k=3) on normalized data : Accuracy = 0.9544, Kappa = 0.3053

SVM on non-normalized data is better on normalized data.
Cross validation on normalized data increases kappa and decreases accuracy a bit. But it is a better trade




Rule Learner Model
```{r}
#Rule learner on non-normalized data
sales_JRips <- JRip(Popularity ~ Year+ Platform + Genre + NA_Sales +Other_Sales+EU_Sales+JP_Sales+Publisher+PlayerType, data = sales_train)
predict_Jrips<-predict(sales_JRips,sales_valid)
summary(predict_Jrips)
confusionMatrix(predict_Jrips,sales_valid$Popularity)

#Rule learner on normalized data
sales_JRips_norm <- JRip(Popularity ~ Year+ Platform + Genre + NA_Sales +Other_Sales+EU_Sales+JP_Sales+Publisher+PlayerType, data = salesnormalized_train)
predict_Jrips_norm<-predict(sales_JRips_norm,salesnormalized_valid)
summary(predict_Jrips)
confusionMatrix(predict_Jrips_norm,salesnormalized_valid$Popularity)
```
Accuracy = 0.9978, Kappa = 0.908 for non-normalized values
Accuracy = 0.9986, Kappa = 0.9311 for normalized values
The rationale behind my above formula is:
- Popularity is derived from Global Sales, and global sales is derived from the sum of JP,NA,Eu and other sales. This createss a model which has exactly spearate classes. Thus leading to overfitting.
- The model has done a pretty good job to read and predict unseen data. Especially, the Kappa value is amazing!

Regression Tree Model
```{r}

rpartsaless <- rpart(Popularity ~ Year+ Platform + Genre + NA_Sales +Other_Sales+EU_Sales+JP_Sales+Publisher + PlayerType, data = salesnormalized_train)
rpartsaless

predictrparts <- predict(rpartsaless, salesnormalized_valid)

summary(predictrparts)
table(predictrparts)
rpart.plot(rpartsaless,fallen.leaves = TRUE, type =1, extra =1, digits=2)

```
There are a lot of takeaways from the rule learner model.
- Popularity of the game is only affected by the sales and the Publisher only.
- The astonishing thing is that the publishers mentioned above (bethesda, EA, take-two interactive, ubisoft etc ) are actually called AAA companies. AAA means the titles were made by an in-house team, under direct control of an experienced producer, with high quality standards and often a new IP or sequel to a previous hit.
- These AAA game companies are the most popular than the Indie game companies mentioned earlier, thus the learner has clearly predicted it.


Creating ensamble model for knn and glm
I have saved the model already for faster training purposes. You just need to run the readRDS() function to read the model saved. This will help the TA and prof to read models faster and not have to train it all over again!
You have to execute ONLY the prediction model and confusion matrix to find the accuracy and kappa
```{r}
# Stacking Algorithms







#stacking algos using all attributes
control <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions = "final", classProbs=TRUE)
algorithmList <- c('knn','rpart')

set.seed(101)
modelswithallattributes <- readRDS("./modelsforallstack.rds")
modelswithallattributes <- caretList(Popularity~ ., data= salesnormalized_train, trControl=control, methodList=algorithmList,metric="Kappa")
modelswithallattributes

#saveRDS(modelswithallattributes, "./modelsforallstack.rds")

stackControlall <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions = "final", classProbs=TRUE)
set.seed(102)
allstacks <- readRDS("./allstacks")
allstacks <- caretStack(modelswithallattributes, method="rf", metric="Kappa", trControl=stackControlall)
allstacks

#saveRDS(allstacks, "./allstacks")

predict_stack_all<-predict(allstacks,salesnormalized_valid)
confusionMatrix(predict_stack_all,salesnormalized_valid$Popularity)

confidenceall <- predict(allstacks, newdata = salesnormalized_valid, interval = 'confidence')
summary(confidenceall)



#stacking algorithms using all non sales attributes
set.seed(101)
modelswithoutsales <- readRDS("./modelsforsomesales.rds")
modelswithoutsales <- caretList(Popularity~ Year+ Platform + Genre +Publisher+PlayerType, data= salesnormalized_train, trControl=control, methodList=algorithmList,metric="Kappa")
modelswithoutsales

#saveRDS(modelswithoutsales, "./modelsforsomesales.rds")

stackControlsome <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions = "final", classProbs=TRUE)
set.seed(102)
somestacks <- readRDS("./somestacks")
somestacks <- caretStack(modelswithoutsales, method="rf", metric="Kappa", trControl=stackControlsome)
somestacks

#saveRDS(somestacks, "./somestacks")

predict_stack_some<-predict(somestacks,salesnormalized_valid)
confusionMatrix(predict_stack_some,salesnormalized_valid$Popularity)

confidencesome <- predict(somestacks, newdata = salesnormalized_valid, interval = 'confidence')
summary(confidencesome)

```
Comparing model accuracies

Accuracy : 0 Kappa : -0.0214 , for all attributes.

Accuracy : 0.9894 Kappa: 0 ,for predictors other than sales.

We chose only normalized values for stacking because we've seen that models work bettter on non normalized values
The stacked model heavily overfits when using all the attributes to predict the popularity of games. This says that too much information to the model can have a negative impact in the analysis.


The stacked model has done a good job of accurately predicting the popularity of unseen data when the attributes other than sales is used. But Kappa is 0 which means that the model is not fully probable enough enough to predict the popularity of unseen data,





Comapring Model Accuracies:
Simple KNN on normalized sales values: Accuracy = 0.9904 , Kappa = 0.1652
Simple KNN on original data set values : Accuracy = 0.9988, Kappa = 0.944
Knn with k-fold cross validation : Accuracy = 0.9904, Kappa = 0.1652
Knn with holdout method evaluation : Accuracy = 0.9904, Kappa = 0.1652 
SVM on normalized data: Accuracy= 0.9894 , Kappa 0
SVM on non nomalized data: Accuracy=1 , Kappa = 1
SVM k -fold closs validation (k=3) on normalized data : Accuracy = 0.9544, Kappa = 0.3053
Rule Learner, Accuracy = 0.9978, Kappa = 0.908 for non-normalized values
Rule Learner, Accuracy = 0.9986, Kappa = 0.9311 for normalized values
Stacked Model, Accuracy : 0 Kappa : -0.0214 , for all attributes.
Stacked Model, Accuracy : 0.9894 Kappa: 0 ,for predictors other than sales.

-From the above, we can say that SVM has done a good job predicting the popularity of unseen video game sales
-But rule learner models and logistic tree regression has shown us trends regrading video game sales.




What I learnt from this:
- I have been following the games industry since 2006 and this project is quite close to my heart. I am a game developer and incorporating machine learning in this project is a great opportunity.
- I have been seeing a lot of trends since these years and this machine learning project has shown me the power of how ML can do wonders.
- With all the data, rule-learners and decision trees have done an amazing job to predict the trends in the gaming sales industry.
- As mentioned before, the models have been showing that AAA games have high sales and plotforms do not matter much.
- It seems that games have to be planned and made, so you'll be asking questions like, what genre, which publisher to chose and what platform. Which in turn acts like a decision trees,

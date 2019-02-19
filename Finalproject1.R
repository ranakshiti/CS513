library(e1071)
library(randomForest)
library(class)
library(tree)
library(leaps)
library(kknn)
library(C50)
library(caret)
library(randomForest)
install.packages("kknn")
install.packages("leaps")
install.packages("xts")
install.packages("zoo")
install.packages("tree")
install.packages("C50")
install.packages("caret")


#Read the data
data <- read.csv("C:/Users/kshit/Documents/Stevens/Semester 2/CS 513 - Knowledge Discovery and Data Mining/Project/Pricedata/PriceData.csv")
View(data)


#Trend generation
dataCopy <- data

library(xts)
dataCopy$Date <- as.Date(dataCopy$Date, format = "%m/%d/%Y")
head(dataCopy$Date)
xyz <- xts(dataCopy$Nifty, order.by = dataCopy$Date)
plot(xyz)

#Standardization of data
for(i in 2:12){
  a <- diff(log(data[,i]))
  a <- c(0,a)
  data[,i] <- a
}
data <- data[-1,]

head(data)


#generate categorical data column
NiftyTrend <- rep("Positive",2174)
NiftyTrend[data$Nifty<0] <- "Negative"
data$NiftyTrend <- NiftyTrend
data$NiftyTrend <- as.factor(data$NiftyTrend)

summary(data)
View(data)
#Generation of training and test dataset  - Modify according to professor

idx<-sort(sample(nrow(data),as.integer(.80*nrow(data))))
train<-data[idx,]
test<-data[-idx,]

cor(train[,-c(1,13)])[11,]

#(1) Simple Linear Regression Model:
s <- regsubsets(Nifty ~ SHCOMP + HangSang + S.P500 +
                  TOPIX + N100 + SPTSX +  KRDOW + TWSE + 
                  ASX + FRDOW, data = train,
                method = "exhaustive")
summary(s)[7]

model1 <- lm(Nifty ~ HangSang, data = train)
summary(model1)

p1 <- predict(model1, newdata=test)

e1 <- mean((test$Nifty - p1)^2)
e1

#(2) Multiple Linear Regression:
c <- summary(s)$cp

plot(c ,
     type='b',
     xlab="No. of Predictors",
     ylab=expression("Mallows C"[P]),
     col="blue")

points(which.min(c),
       c[which.min(c)],
       pch=20,
       col="red")

summary(s)[7]

model2 <- lm(Nifty ~ HangSang + TOPIX + N100 + KRDOW, data = train)

summary(model2)

#removing TOPIX
model2 <- lm(Nifty ~ HangSang + N100 + KRDOW, data = train)

#predicting using the model2
p2 <- predict(model2, newdata=test)

e2 <- mean((test$Nifty - p2)^2)
e2
summary(model2)

#support vector machine
model3 <- svm(Nifty ~ HangSang + TOPIX + N100 + KRDOW, data = train,
              type = "eps-regression")
p3 <- predict(model3, newdata=test)

e3 <- mean((test$Nifty - p3)^2)
e3

#(4) Random Forest Regression:
model4 <- randomForest(x = train[,c(3,5,6,8)], y = train$Nifty, ntree = 501)
p4 <- predict(model4, newdata=test)



e4 <- mean((test$Nifty - p4)^2)
e4

#KNN:
trainKNN <- as.matrix(data.frame(train$HangSang, train$TOPIX, 
                                 train$N100, train$KRDOW))
testKNN <- as.matrix(data.frame(test$HangSang, test$TOPIX, 
                                test$N100, test$KRDOW))
dirTrain <- train$NiftyTrend
#Trying KNN from k=1 to k=50
dirTest <- test$NiftyTrend
e44 <- NULL
for(i in 1:50){
  set.seed(1002)
  pKNN <- knn(trainKNN,testKNN,
              dirTrain,k=i)
  e44[i] <- mean(pKNN != dirTest) # Misclassification Error
}
plot(e44,
     type='b',
     xlab="k",
     ylab="Error",
     col="blue")
points(which.min(e44),
       e44[which.min(e44)],
       pch=20,
       col="red")

kselected <- which.min(e44)

#We get minimum misclassification error at k = 46. So, we build the final KNN model with k=36.
m4 <- knn(trainKNN, testKNN,
          dirTrain, k = kselected)

#Confusion Matrix:
knn_table<-table(m4, dirTest)
cm_knn<-confusionMatrix(knn_table)
acc_knn <- cm_knn$byClass['Accuracy']
acc_knn
#Misclassification Error:

e444 <- mean(m4!=dirTest)
e444

#(5) Random Forest Tree:
library(randomForest)
set.seed(101)

m5 <- randomForest(NiftyTrend ~ HangSang + TOPIX + N100 + KRDOW,
                   data=train, importance=TRUE)
m5 
p55 = predict(m5, newdata=test, type="class")
#Confusion Matrix:
table(p55, dirTest)

#Misclassification Error:
e55 <- mean(p55 != dirTest)
e55

#tree
tree <- tree(NiftyTrend ~ HangSang + TOPIX + N100 + KRDOW,
             data=train)
summary(tree)

plot(tree)
text(tree,pretty=0) # Plotting Tree

p66 <- predict(tree, newdata=test, type="class")

#Confusion Matrix:
table(p66, dirTest)
#Misclassification Error:
e66 <- mean(p66 != dirTest)
e66

#naive bayes
nBayes_all <- naiveBayes(NiftyTrend~., data = data)

category_all<-predict(nBayes_all,data)

table(NBayes_all=category_all,NiftyTrend=data$NiftyTrend)
NB_wrong<-sum(category_all!=data$NiftyTrend)
NB_error_rate<-NB_wrong/length(category_all)
NB_error_rate

#KKNN doubt
?kknn()

predict_k5 <- kknn(formula = NiftyTrend~., train, test, k = kselected, kernel = "triangular")
fit <- fitted(predict_k5)
table(test$NiftyTrend,fit)
kknnrate <- mean(fit!=dirTest)
kknnrate

#ann
library("neuralnet")
set.seed(321)
ann <- neuralnet(NiftyTrend ~ HangSang + TOPIX + N100 + KRDOW,trainKNN,hidden = 5, threshold = 0.01)

#Summary for all models
tab <- data.frame("Methods" = c("Naive Bayes","KNN","KKNN","Tree", "Random Forest Tree"), 
                  "Misclassification Errors" = 
                    c( NB_error_rate,e444,kknnrate, e55, e66))

tab
?svm()

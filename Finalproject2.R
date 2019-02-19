library(caret)

rm(list=ls())

#Read the pricedata set
price_data <- read.csv("C:/Users/kshit/Documents/Stevens/Semester 2/CS 513 - Knowledge Discovery and Data Mining/Project/PriceData.csv")
View(price_data)

#Standardization of data
for(i in 2:12){
  a <- diff(log(price_data[,i]))
  a <- c(0,a)
  price_data[,i] <- a
}
price_data <- price_data[-1,]

#generate categorical data column
NiftyTrend <- rep("Positive",2174)
NiftyTrend[price_data$Nifty<0] <- "Negative"
price_data$NiftyTrend <- NiftyTrend
price_data$NiftyTrend <- as.factor(price_data$NiftyTrend)
summary(price_data)
View(price_data)

#Generation of training and test dataset
idx<-sort(sample(nrow(price_data),as.integer(.80*nrow(price_data))))
train<-price_data[idx,]
test<-price_data[-idx,]

#Finding co-relation between columns
cor(train[,-c(1,13)])[11,]

#KNN:
library(class)
trainKNN <- as.matrix(data.frame(train$HangSang, train$TOPIX, 
                                 train$N100, train$KRDOW))
testKNN <- as.matrix(data.frame(test$HangSang, test$TOPIX, 
                                test$N100, test$KRDOW))
train_label <- train$NiftyTrend
test_label<- test$NiftyTrend

#Trying KNN from k=1 to k=50
e1 <- NULL
for(i in 1:50){
  set.seed(1002)
  pKNN <- knn(trainKNN,testKNN,
              train_label,k=i)
  e1[i] <- mean(pKNN != test_label) # Misclassification Error
}
plot(e1,
     type='b',
     xlab="k",
     ylab="Error",
     col="blue")
points(which.min(e1),
       e1[which.min(e1)],
       pch=20,
       col="red")

k_min <- which.min(e1)

#We get minimum misclassification error at k = 27. So, we build the final KNN model with k=27.
knn_m <- knn(trainKNN, testKNN,
          train_label, k = k_min)

#Confusion Matrix:
knn_table<-table(knn_m, test_label)
cm_knn<-confusionMatrix(knn_table)
acc_knn <- cm_knn$overall['Accuracy']
#Accuracy for KNN
acc_knn

#support vector machine
library(e1071)
?svm()
svm_model <- svm(Nifty ~ HangSang + TOPIX + N100 + KRDOW, data = train,
              type = "eps-regression")
svm_p <- predict(svm_model, newdata=test)

#Random Forest
library(randomForest)
set.seed(101)

rf <- randomForest(NiftyTrend ~ HangSang + TOPIX + N100 + KRDOW,
                   data=train, importance=TRUE)
rf 
rf_p<- predict(rf, newdata=test, type="class")
#Confusion Matrix:
rf_table <- table(rf_p, test_label)
cm_rf <- confusionMatrix(rf_table)
acc_rf <- cm_rf$overall['Accuracy']
acc_rf

#naive bayes
library(e1071)
nBayes_all <- naiveBayes(NiftyTrend~., data = price_data)
category_all<-predict(nBayes_all,price_data)

nb_table <- table(NBayes_all=category_all,NiftyTrend=price_data$NiftyTrend)
cm_nb <- confusionMatrix(nb_table)
acc_nb <- cm_nb$overall['Accuracy']
acc_nb

#tree
library(tree)
tree <- tree(NiftyTrend ~ HangSang + TOPIX + N100 + KRDOW,
             data=train)
summary(tree)

plot(tree)
text(tree,pretty=0) # Plotting Tree

tree_p <- predict(tree, newdata=test, type="class")
tree_table <- table(tree_p,test_label)

cm_tree <- confusionMatrix(tree_table)
acc_tree <- cm_tree$overall['Accuracy']
acc_tree

#KKNN
library(kknn)
kknn_p <- kknn(formula = NiftyTrend~., train, test, k = k_min, kernel = "triangular")
fit <- fitted(kknn_p)
kknn_table <- table(test$NiftyTrend,fit)
cm_kknn <- confusionMatrix(kknn_table)
acc_kknn <- cm_kknn$overall['Accuracy']
acc_kknn

#Summary for all models
summary_table <- data.frame("Methods" = c("Naive Bayes","KNN","KKNN","Tree", "Random Forest"), 
                  "Accuracy" = c(acc_nb,acc_knn,acc_kknn,acc_tree,acc_rf))

summary_table

#Trend generation
price_d <- price_data

library(xts)
price_d$Date <- as.Date(price_d$Date, format = "%m/%d/%Y")
head(price_d$Date)
trend <- xts(price_d$Nifty, order.by = price_d$Date)
plot(trend)


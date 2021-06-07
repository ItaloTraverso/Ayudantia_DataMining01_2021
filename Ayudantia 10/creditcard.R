library(tidyverse)
library(e1071)
library(caret)
library(rstan)
library(rstanarm)

setwd("D:/Users/Italo/Documents/Italo Felipe/UAI/Semestre 11/Ayudantia Mineria de Datos/material ayudantia/Ayudantia10")

creditcard <- read.csv("UCI_Credit_Card.csv")

glimpse(creditcard)

str(creditcard)

creditcard$ID <- NULL

creditcard$SEX <- factor(creditcard$SEX, levels=1:2, labels=c("Male", "Female"))
creditcard$EDUCATION <- as.factor(creditcard$EDUCATION)
creditcard$MARRIAGE <- as.factor(creditcard$MARRIAGE)
creditcard$default.payment.next.month <- factor(creditcard$default.payment.next.month, levels = 0:1, labels=c("No", "Yes"))

set.seed(42)
sample <- sample(1:nrow(creditcard), .8*30000)

trainData <- creditcard[sample,]
testData <-  creditcard[-sample,]

model_nb <- naiveBayes(default.payment.next.month~SEX+MARRIAGE+AGE+PAY_0+PAY_6+BILL_AMT1+BILL_AMT6+PAY_AMT1+PAY_AMT6, trainData, laplace=1)

pred_nb <- predict(model_nb, newdata = testData)
confusionMatrix(data=pred_nb, reference = testData$default.payment.next.month)

library(ROCR)

pred_test_nb <- predict(model_nb, newdata = testData, type="raw")
p_test_nb <- prediction(pred_test_nb[,2], testData$default.payment.next.month)
perf_nb <- performance(p_test_nb, "tpr", "fpr")
plot(perf_nb, colorize=T)
performance(p_test_nb, "auc")@y.values
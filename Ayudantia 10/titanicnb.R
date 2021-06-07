library(tidyverse)
library(e1071)
library(caret)
library(rstan)
library(rstanarm)
library(titanic)

#mybayesreg <- stan_glm(y ~ X1 + x2 + x3 ..., 
#                      family = myfamily, data = mydata, 
#                       prior = myprior)

titanictrain <- titanic::titanic_train %>%  as.data.frame()
titanictest <- titanic::titanic_test %>% as.data.frame()

titanic <- titanictrain

# Display titanic data
glimpse(titanic,width = 50)

# Reformat Class
titanic$class <- str_extract(titanic$Pclass, "[0-9]")
titanic$SexCode <- (titanic$Sex == "female") %>% as.numeric()

titanic <- titanictrain[c(4,3,5:12,2)]

titanic$PassengerId <- NULL
titanic$Sex <- NULL
titanic$Ticket <- NULL
titanic$Cabin <- NULL
titanic$SibSp <- NULL
titanic$Parch <- NULL
titanic$Fare <- NULL
titanic$Embarked <- NULL
titanic$Pclass <- NULL

TitanicLinear <- stan_glm(Survived ~ Age + SexCode + as.factor(class), 
                          data = titanic, family = gaussian)

summary(TitanicLinear)

posterior_interval(TitanicLinear, prob=0.95)

plot(TitanicLinear)

Titanic_posterior <- TitanicLinear %>% as_tibble() %>% 
  rename(sec.class = "as.factor(class)2",
         third.class = "as.factor(class)3")

ggplot(Titanic_posterior, aes(x=third.class)) + 
  geom_histogram()

posterior_vs_prior(TitanicLinear)

pp_check(TitanicLinear)

#----------------------------------------------------------------------
# Analysis how the classification error of a bagging ensemble
# varies with the size of the ensemble considering trees and logit
# as base learners
#----------------------------------------------------------------------
# Libraries needed in the demo (see exercise 5 for details)
library(rpart)

# Load and briefly prepare the data
uni <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv", 
                header=TRUE, sep=",")
uni$rank  <- factor(uni$rank, levels = c(4,3,2,1))
uni$admit <- factor(uni$admit)

#----------------------------------------------------------------------
# Helper functions 
#----------------------------------------------------------------------


# Helper to calculate classification error based on probabilistic 
# model predictions.
# We once agin set the  classification threshold such that the 
# number of predicted positives and negatives mimics the prior 
# probability of these classes in the training set
class.prediction <- function(yhat, frac.pos) {
  n <- floor(length(yhat)*frac.pos)
  sort.idx <- sort(yhat, index.return=TRUE, decreasing = TRUE)$ix[1:n]
  c <- rep(0, length(yhat))
  c[sort.idx]=1
  return(c)
}

# Helper function to compute composite forecast of an ensemble 
# for a specified data set called <data>
composite.forecast <- function(bagger, data) {
  t <-length(bagger)
  
  for (i in 1:t) {
    # call prediction function for logit or tree
    if (class(bagger[[i]])[1]=="rpart") {
      tmp <- predict(bagger[[i]], newdata=data)[,2] 
    }
    else {
      tmp <- predict(bagger[[i]], newdata=data, type="response")  
    }
    # store forecasts
    if (i==1) 
      yhat <- data.frame(V1=tmp)
    else 
      yhat[,i] <- tmp
  }
  # average probabilistic ensemble predictions
  yhat.ensemble <- rowMeans(yhat)
  return(yhat.ensemble)
}

# Helpfer function to compute the classification error given some 
# predictions and corresponding test data, where we assume a 
# binary target variable called <admit>
classification.error <- function(yhat, data, prior.class1) {
  
  # check if predictions are categorical (only a crude test)
  if ( length(unique(yhat))>2 ) {
    class.pred <- class.prediction(yhat, prior.class1) 
  } else {
    class.pred <- yhat  
  }
    
  # factorize class predictions
  class.pred <- factor(class.pred, levels = c(0,1), labels = c(0,1))  
  # calculate classification error 
  cerr <- sum(data$admit != class.pred) / nrow(data)
  return(cerr)
}

#----------------------------------------------------------------------
# Main part of the Demo
#
# First configure the size of the bagging ensemble. Next create ensemble
# models using trees and logit as base learner. For each ensemble, 
# calculate the classification error and trace it's development over the
# number of bootstrap iterations; that is ensemble sizes 
#----------------------------------------------------------------------

# Number of iterations
T = 1000 # be careful with settings of T. Values above 100 can lead to long runtimes
# Allocate memory for tree/logit ensemble
bagger.tree  <- list(T)
bagger.logit <- list(T)

set.seed(123) # for replication

# Data partitioning:
# In this example we need to draw a fixed hold-out test set up front.
# Ohterwise, we could not trace the development classification error on
# the same data.
frac <-0.7
idx <- sample(nrow(uni), floor(nrow(uni)*frac), replace=FALSE)
data.train <- uni[ idx,]
data.test  <- uni[-idx,]

# Grow ensemble model of size T
for (i in 1:T) {
  n<-nrow(data.train)
  bts <- data.train[sample(n, n,  replace=TRUE),]
  bagger.tree[[i]]  <- rpart(admit ~., data = bts)
  bagger.logit[[i]] <- glm(admit ~.,   data = bts, family = binomial(link = logit))
}
#----------------------------------------------------------------------
# Collected data for plotting

p.admit <- sum(data.train$admit==1)/nrow(data.train) # needed to binarize forecasts
error.trace <- data.frame(Iter=1:T, Logit=rep(1,T), Tree=rep(1,T))
for (i in 1:T) {
  # Calculate composite forecast of an ensemble with i members. Note how it
  # is essential to index with 1:i in the following row. What would be the 
  # result of just using bagger.tree without index as input to the function?
  yhat <- composite.forecast(bagger.tree[1:i], data=data.test)
  error.trace$Tree[i] <- classification.error(yhat, data=data.test, prior.class1=p.admit)
  
  # Repeat the above for the ensemble of logit models (just in one call)
  error.trace$Logit[i] <- classification.error( composite.forecast( bagger.logit[1:i], 
                                                              data=data.test), 
                                          data=data.test, 
                                          prior.class1=p.admit)
}
#----------------------------------------------------------------------
# Plot classification error development as a line plot
library(ggplot2)
n<- nrow(error.trace) # set smaller value to focus plot
ggplot(error.trace[,] , aes(x=Iter)) + geom_line(aes(y=Tree, color='green')) +
  geom_line(aes(y=Logit, color='red')) + 
  labs(x='bagging iterations', y='classification error', color="Base\nlearner") +
  scale_color_manual(labels = c("Tree", "Logit"), values = c("blue", "red")) +
  theme(legend.position="right", legend.box='vertical')

#----------------------------------------------------------------------
# DEMO Implementation of Bagging Algorithm in R
#----------------------------------------------------------------------
# Libraries needed in the demo (see exercise 5 for details)
library(rpart)

# Load and briefly prepare the data
uni <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv", 
                header=TRUE, sep=",")
uni$rank  <- factor(uni$rank, levels = c(4,3,2,1))
uni$admit <- factor(uni$admit)
#----------------------------------------------------------------------
# Data Partitioning:
#   randomly split data into 70% training and 30% test
#   control random number generation to ensure replicability
set.seed(123) # to ensure replicability
idx <- sample(nrow(uni), floor(nrow(uni)*.7), replace=FALSE)
tr <-uni[idx,]
ts <-uni[-idx,]

#----------------------------------------------------------------------
# Set size of ensemble 
T=10
# Reserve memory for ensemble members
tree.bagger  = list(T)
#----------------------------------------------------------------------
# Ensemble creation loop
for (i in 1:T) {
  # draw bootstrap sample
  bts = sample(nrow(tr), nrow(tr), replace = TRUE)
  # estimate tree/logit model on the sample
  tree.bagger[[i]] <- rpart(admit ~., data = uni[bts,])
}
#----------------------------------------------------------------------
# Apply ensemble to test data:
#   reserve memory to store forecasts
tree.forecasts <- data.frame(rep(0,nrow(ts)))
for (i in 1:T) {
  tmp = predict(tree.bagger[[i]], newdata = ts)[,2]
  tree.forecasts[,i]<-tmp  
}

# obtain ensemble prediction through averaging
yhat <- rowMeans(tree.forecasts[,2:(T+1)])


# calculate Brier Score
bs.tree<-1/nrow(ts)*sum(((as.numeric(ts$admit)-1)-yhat)^2)
bs.tree

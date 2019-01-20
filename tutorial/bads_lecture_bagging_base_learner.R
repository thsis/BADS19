#----------------------------------------------------------------------
# Analysis of using bagging with different base learners
#----------------------------------------------------------------------
# Libraries needed in the demo (see exercise 5 for details)
library(rpart)

# Load and briefly prepare the data
uni <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv", 
                header=TRUE, sep=",")
uni$rank  <- factor(uni$rank, levels = c(4,3,2,1))
uni$admit <- factor(uni$admit)
#----------------------------------------------------------------------
# Later we will calculate classification error. To that end, we require
# a classification threshold, which we will set according to the class
# distribution in the data
prior.class1 <- sum(uni$admit==1)/nrow(uni)

# Helper function to convert probabilistic forecasts into crisp classifications
class.prediction <- function(yhat, frac.pos) {
  n <- floor(length(yhat)*frac.pos)
  sort.idx <- sort(yhat, index.return=TRUE, decreasing = TRUE)$ix[1:n]
  c <- rep(0, length(yhat))
  c[sort.idx]=1
  return(c)
}

# Helper function to compute classification error of a classification tree
calc.error.tree <- function(tr, ts, prior.class1) {
  tree <- rpart(admit ~., data = tr)
  yhat <- predict(tree, newdata= ts)[,2]
  yclass <- class.prediction(yhat, prior.class1) 
  cerr <- sum( ts$admit != factor(yclass, levels = c(0,1), labels = c(0,1)))/nrow(ts)
  return(cerr)
} 

# Helper function to compute classification error of a logit model
calc.error.logit <- function(tr, ts, prior.class1) {
  logit <- glm(admit ~., data = tr, family = binomial(link = logit))
  yhat <- predict(logit, newdata= ts, type="response")
  yclass <- class.prediction(yhat, prior.class1) 
  cerr <- sum( ts$admit != factor(yclass, levels = c(0,1), labels = c(0,1)))/nrow(ts)
  return(cerr)
} 

#----------------------------------------------------------------------
# Number of iterations
T = 100 # set this parameter to control the number of iterations
cerrors <- data.frame(Tree=rep(0, T), Logit=rep(0, T))
set.seed(123)
for (i in 1:T) {
  #----------------------------------------------------------------------
  # Data Partitioning:
  idx <- sample(nrow(uni), nrow(uni), replace=TRUE)
  oob <- uni[setdiff(1:nrow(uni), unique(idx)), ]
  cerrors$Tree[i] <- calc.error.tree(uni[idx,], oob, prior.class1)
  cerrors$Logit[i] <- calc.error.logit(uni[idx,], oob, prior.class1)
}
#----------------------------------------------------------------------
# Boxplot of results
library(ggplot2)
library(reshape2)
data <- data.frame( Model=factor(rep(c("Tree","Logit"), each=nrow(cerrors))), 
                    CErr=c(cerrors$Tree, cerrors$Logit))
# note that the above reaggangement of the data.frame could easily be 
# performed by using the melt() function from the reshape2 package.
# For example, the call data<-melt(cerrors) produces the same result

# Boxplot
ggplot(data , aes(x=Model, y=CErr, fill=Model)) + geom_boxplot() + 
  labs(x='base learner', y='classification error') +
  theme(legend.position="none", 
        axis.text=element_text(size=14),
        axis.title=element_text(size=16,face="bold"))


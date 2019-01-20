# 1: read data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = read.csv(url, sep = ";")

# 2: first look and cleaning
dim(wine) 
sum(is.na(wine))
# wine$quality = factor(wine$quality)

# 3: bivariate plots
target = wine$quality
features = wine[, -which(colnames(wine) == "quality")]

sapply(colnames(features), function(x) plot(y = target, x = features[, x], xlab = x))

# 4: correlation plots
corrplot::corrplot(cor(features))

# 5: regression with 2 most important variables (highest correlation with quality?)
model1 = lm("quality ~ volatile.acidity + alcohol", data = wine)
summary(model1)

# 6: regression with all variables
model2 = lm("quality ~ .", data = wine)
summary(model2)


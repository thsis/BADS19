# k-means depends on random initialization - so just specify a seed.
set.seed(42)
source("Rintro_HelperFunctions.R")

loans = GetLoanDataset()
idx_numeric =  which(sapply(X = loans, FUN = is.numeric))

data = scale(loans[, idx_numeric])

results = kmeans(data, 5, iter.max = 50)
clusters = results$cluster

k.settings = 1:15
obj.settings = vector(mode = "integer", length = length(k.settings))

for (i in k.settings) {
  clu.sol = kmeans(data, k.settings[i], iter.max = 50, nstart=25)
  obj.settings[i] = clu.sol$tot.withinss
}

plot(k.settings, obj.settings, type="l",
     main = "Elbow curve for k selection",
     xlab = "k", ylab = "Total within-cluster SS")
points(k.settings, obj.settings)


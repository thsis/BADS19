# Variables and Classes
# 1.
a <- 3
b <- 4.5

# 2.
class(a)

# 3.
class(a) == "character"

# 4.
a^2 + 1/b
sqrt(a * b)
log(a, base = 2)

# Matrix algebra
A <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 10),
            nrow = 3,
            ncol = 3,
            byrow = TRUE)
B <- matrix(1:9, nrow=3, ncol=3)
y <- 1:3

# 1.
a * A

# 2.
A %*% B

# 3.
invA <- solve(A)

# 4.
A %*% invA

# 5.
t(B)

# 6.
B[1, ] <- 1

# 7.
solve(t(A) %*% A) %*% t(A) %*% y

# Indexing
# 1.
A
B
y

# 2.
A[3, 2] * B[2, 1]

# 3.
A[1, ] * B[, 3]

# 4.
y[y > 1]

# 5.
A[, 2][A[, 1] >= 4]

# 6.
# A[, 4]
# subscript out of bounds: A only has three columns

# Custom functions
standardize <- function(x){
  if(is.numeric(x)){
    avg = mean(x)
    std = sd(x)
    return((x - avg) / std)
  } else return(x)
}

test <- c(-100, -25, -10, 0, 10, 25, 100)
standardize(test)

# Using built-in functions
# 1.
x <- seq(-3, 3, 0.2)
nvValues <- dnorm(x)
plot(x, nvValues, type="l", col="red")

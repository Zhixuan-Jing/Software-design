library("bzinb")
library("stats")
library("Matrix")
# Simulator for spatial transcriptomic data baseline
# input:
# xvec: marginal vector x
# yvec: marginal vector y
# r: size of 2d array (r x r)
# output: simulate 2d array res
simulate <- function(xvec, yvec) {
  fit <- bnb(
  xvec = xvec,
  yvec = yvec,
  em = TRUE,
  tol = 1e-08,
  maxiter = 50000,
  vcov = TRUE,
  initial = NULL,
  showFlag = FALSE
)
res <- fit$coefficients
r <- length(xvec)

data <- rbnb(r + 3, res[1], res[2], res[3], res[4], res[5])
x <- data[, 1]
y <- data[, 2]

res <- x %o% y
print(res)
samples <- sample(x = 1:30, size = 5, replace = TRUE, prob = NULL)
s1 <- res[c(samples[1]:(samples[1] + length(xvec)-1)), c(samples[1]:(samples[1] + length(yvec)-1))]
s2 <- res[c(samples[2]:(samples[2] + length(xvec)-1)), c(samples[2]:(samples[2] + length(yvec)-1))]
s3 <- res[c(samples[3]:(samples[3] + length(xvec)-1)), c(samples[3]:(samples[3] + length(yvec)-1))]
s4 <- res[c(samples[4]:(samples[4] + length(xvec)-1)), c(samples[4]:(samples[4] + length(yvec)-1))]
s5 <- res[c(samples[5]:(samples[5] + length(xvec)-1)), c(samples[5]:(samples[5] + length(yvec)-1))]
g <- (s1 + s2 + s3 + s4 + s5) / 5
return(g)
}
data1 <- rnbinom(30, 2, 0.8)
data2 <- rnbinom(30, 2, 0.5)
print("start")
a <- simulate(data1, data2)
print(a)
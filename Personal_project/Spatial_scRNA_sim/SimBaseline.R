set.seed(2)
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

data <- rbnb(r + 30, res[1], res[2], res[3], res[4], res[5])
x <- data[, 1]
y <- data[, 2]

res <- sqrt(x %o% y)

return(res)
}

# Testing
# data1 <- rnbinom(30, 2, 0.8)
# data2 <- rnbinom(30, 2, 0.5)
# print("start")
# a <- simulate(data1, data2)
# print(a)
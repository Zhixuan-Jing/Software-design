library("bzinb")
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
r <- max(length(xvec), length(yvec))
data <- rbnb(r, res[1], res[2], res[3], res[4], res[5])
x <- data[, 1]
y <- data[, 2]

res <- x %o% y
return(res)
}
data1 <- rnbinom(10, 1, 0.2)
data2 <- rnbinom(10, 1, 0.5)
print(data1)
a <- simulate(data1, data2)
print(a)
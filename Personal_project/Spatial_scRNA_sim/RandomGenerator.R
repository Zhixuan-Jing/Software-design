set.seed(1)
library("Matrix")
library("stats")
data1 <- rnbinom(10, 5, 0.7)
data2 <- rnbinom(8, 4, 0.5)

a <- data1 %o% data2

feat_extraction <- function(a) {
  xvec <- rowMeans(a)
  yvec <- colMeans(a)
  return(list("vx" = xvec, "vy" = yvec))
}

r <- feat_extraction(a)
x <- r$vx
y <- r$vy
print(x)
print(y)
set.seed(1)

data1 <- rnbinom(10, 5, 0.7)
data2 <- rnbinom(8, 4, 0.5)

a <- data1 %o% data2
print(a)
xvec <- rowMeans(a)
yvec <- colMeans(a)
print(length(xvec))
print(length(yvec))

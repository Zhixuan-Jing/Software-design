compare <- function(x1, x2) {
  dim(x1) <- c(1, length(x1))
  dim(x2) <- c(1, length(x2))
  x1 <- as.vector(x1)
  x2 <- as.vector((x2))
  print("line 6")
  boxplot(x1, x2, names = c("x1", "x2"), col = colors()[10:11])
}

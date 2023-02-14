
library("Matrix")
library("stats")

# nb random st generator for generating original ST data
# input: parameter vector s1 and s2
# output: random st-based image a
nb_random <- function(s1, s2) {
  data1 <- rnbinom(s1[1], s1[2], s1[3])
  data2 <- rnbinom(s2[1], s2[2], s2[3])

  a <- data1 %o% data2
  return(a)
}

# feature extraction for existed data
# input: st data matrix a
# output: row and column mean vectors xvec and yvec

feat_extraction <- function(a) {
  xvec <- rowMeans(a)
  yvec <- colMeans(a)
  return(list("vx" = xvec, "vy" = yvec))
}
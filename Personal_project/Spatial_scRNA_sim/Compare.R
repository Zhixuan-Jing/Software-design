# The compare methods of spatial data simulation
# The input is uniformed as (x1, x2: matrix, matrix) as the candidate and standard

c_stats <- function(x1, x2) {
  dim(x1) <- c(1, length(x1))
  dim(x2) <- c(1, length(x2))
  x1 <- as.vector(x1)
  x2 <- as.vector((x2))
  boxplot(x1, x2, names = c("candidate", "standard"), col = colors()[10:11])
}

c_spat_mse <- function(x1, x2) {
  mse <- (x1 - x2) * (x1 - x2)

  return(mse)

}

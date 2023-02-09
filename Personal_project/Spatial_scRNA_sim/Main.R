# This program is to test all functions without generating R package
# Author: Jing Zhixuan
# Date: 07.02.2023
source("e:\\Software projects\\Software-design\\Personal_project\\Spatial_scRNA_sim\\SimBaseline.R")
source("e:\\Software projects\\Software-design\\Personal_project\\Spatial_scRNA_sim\\RandomGenerator.R")
source("e:\\Software projects\\Software-design\\Personal_project\\Spatial_scRNA_sim\\Compare.R")

# Generating random variables


s1 <- c(100, 4, 0.5)
s2 <- c(100, 3, 0.5)
print("generating random data...")
a <- nb_random(s1, s2)
print("Random data generation completed")

print("Simulation start...")
v <- feat_extraction(a)
xvec <- v$vx
yvec <- v$vy
# c_stats(sqrt(xvec %o% yvec), a)
res <- simulate(xvec, yvec)
# print(res)
# print("start evaluation...")
print("statistical error:")
c_stats(res, a)
# print("spatial MSE: ")
# print(c_spat_mse(res, a))
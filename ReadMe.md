The MATLAB code package implements the stochastic variational inference algorithm to learn the ensemble of latent trees for spatial extremes as described in the following paper:

H. Yu, and J. Dauwels, ''Latent tree ensemble of pairwise copulas for spatial extremes analysis,'' to be appear in IEEE Trans. Signal Process.

The main script is main.m. To test your own datasets, please follow the steps below:

1. Replace the artifical data "MS8x8" by your own block maxima data set when loading data and store the data in the nxp matrix XDat, where n is the no. of time series and p is the no. of locations.  

2. Please input the values of pc and pr, where pr and pc are respectively the no. of rows and columns of the lattice where we observe the block maxima (i.e., the bottom scale of the pyramidal model).


Yu Hang, NTU, Jan 2016.
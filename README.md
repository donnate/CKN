# CKN


This implements in pytorch the Convolutional Kernel Approach proposed by Mairal and al in their 2014 NIPS paper [__Convolutional Kernel Networks__](https://arxiv.org/abs/1406.3332).


Some small modifications were made with respect to the original paper (see below). 
In short, the training algorithm works as follows:


+ (1) extract patches from input map (by concatenating neighboring pixel input). Normalize the patches so that they all have unit norm. (Keep the orginal norms in memory, in Cell.norms).
+ (2) apply dimensionality reduction to these large patches (RobustScaler+PCA), so that the whole thing remains computable
+ (3) initalize new filters with Kmeans
+ (4) train W and eta (parameters of the cell) on a subset of the data. The function check_convergence() allows to check how well we are doing in terms of approximating the kernel.
+ (5) call get_activation_map() to compute the activation of each patch

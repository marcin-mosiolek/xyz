# Autencoder
Please find below, a short description of the most interesting scripts in the repository.

---
`extract_clusters.py`: this is basically the same script as the one delivered with the dataset, with simple modification: it extracts the clusters and saves them as a list of frames, where each frame consits of a list of clusters.

You may need to modify following variables in the script:
* `OUTPUT` = a path where the result should be stored
* variables from the original script indicating where to look for the dataset

---
`convex_hulls.py`: in order to create the training set for autoencoder, we need to create a list of pairs `(x, y)` where `x` corresponds to raw points projection on 2D grid and `y` is the grid with plotted ground truth convex hulls. This representation is then used by the training script to "teach" the autencoder to convert 2D grid to representation similar to convex hulls, which is then used for CCA clustering. This script maintains the information about the original cluster ids, as it is then exploited in `eval.py` and `compare.py` for evaluating adjusted mutual information score.

You may need to modify following variables in the script:
* `INPUT` = a path to the result file of `extract_clusters.py`
* `OUTPUT` = a path where the result should be stored

----

`train.py`: as the name suggests the script trains the model and plots `MSE` results for training and validation set after each epoch. Trained model is saved as `conv_autoencoder.pth`. The script uses early stopping if the validation loss doesn't improve within 5 epochs and saves only the best model. 

You may need to modify following variables in the script:
* `INPUT` = a path to the result file of `convex_hulls.py`

The MSE loss takes very small values, which might be partially explained by the fact the grids are made mostly of zeros and MSE by default averages the values. You may try to train the network as well with the parameter `size_average=False` for the `MSELoss`. Other issues, which I may see, are addressed int the TODO section.

--- 

`eval.py`: the script runs the evaluation of the baseline algorithm vs autencoder algorithm and calculates `adjusted mutual information score`. The result is then stored as `eval.csv`. 

You may need to modify following variables in the script:
* `INPUT` = a path to the result file of `convex_hulls.py`

The baseline algorithm is executed on the full 2D grid. To make the comparison fair, the `adjusted mututal information score` is calculated only for the points which are labelled both by the baseline and autoencode/convex hulls algorithms. Also simple analysis of how many clusters were missed by the convex hulls reconstruction is printed out. You may notice that:
* ~40% of the predictions contains the same number of convex hulls as in the ground truth
* ~35% of the predictions misses/ads 1 convex hulls
* ~20% of the predictions misses/ads 2 convex hulls
* ~5% of the predictions misses/ads >2 convex hulls

---
`tools.py`: just a helper script, which loads the data and provides pytorch functions wrappers.

### In progress
- the model doesn't handle well small convex hulls, this might be related also to the quality of the training set

### TODO
- the model is the first thing, that came to my mind, it may have to big capacity for current dataset
- validation set is too similar to training set
- no augmentation of the dataset was made
- the more lidar frames we have, the better
- the grid representation is super redundant as it contains mostly of zeros

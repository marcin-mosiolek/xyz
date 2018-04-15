
import torch
import numpy as np
from sklearn import metrics

from torch.autograd import Variable
from torch import nn

from tools import DataLoader
from model import AutoEncoder
from tools import make_gpu, make_numpy
from scipy import ndimage

from progress.bar import Bar

import time


def stats(name, data):
    print("\n============ {} \n============".format(name))
    print("Mean: {}".format(np.mean(data)))
    print("Std: {}".format(np.mean(data)))
    q = [x for x in np.linspace(0, 100, 21)]
    percentiles = np.percentile(data, q=q)
    print("Percentiles:")
    for i, p in zip(q, percentiles):
        print("{}%:{:.3f}".format(i, p))


def extract_no_clusters(grid):
    # -1 because it returns 0 as well
    return len(np.unique(grid) - 1)


def main():
    # load data
    data = DataLoader("../autencoder/convex_hulls.npy")

    # load the model and parameters
    model = AutoEncoder().cuda()

    # load the weights from file
    weights = torch.load("./conv_autoencoder.pth")
    model.load_state_dict(weights)

    # store stats
    exec_times = []
    no_true_clusters = []
    no_predicted_clusters = []
    score = []

    progress = Bar("Eval", max=len(data.valid_x))
    for grid, x, y in zip(data.x, data.valid_x, data.valid_y):
        progress.next()

        # measure execution time of clustering
        start_time = time.time()
        x = x.reshape(-1, 1, 300, 400)
        predicted_y = model(make_gpu(x))
        predicted_y = make_numpy(predicted_y.cpu())
        predicted_y = predicted_y.reshape(300, 400)
        predicted_grid, pred_no_clusters = ndimage.label(predicted_y, structure=np.ones((3, 3)))
        end_time = time.time()

        # measure the clustering quality
        print(grid.shape)
        valid_inds = np.where((x > 0) & (predicted_grid > 0))[0]
        true_labels = x[valid_inds]
        print(true_labels.shape)
        predicted_labels = predicted_grid[valid_inds]
        score.append(metrics.adjusted_rand_score(true_labels, predicted_labels))

        # log the stuff
        no_true_clusters.append(extract_no_clusters(grid))
        no_predicted_clusters.append(pred_no_clusters)
        exec_times.append(end_time - start_time)
    progress.finish()

    stats("Execution time", exec_times)
    stats("Clusters no", np.array(no_true_clusters) - np.array(no_predicted_clusters))
    stats("Adjusted rand score", score)


if __name__ == "__main__":
    main()
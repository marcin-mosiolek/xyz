
import torch
import numpy as np
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
    return len(np.unique(grid))


def main(batch_size = 64):
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

    progress = Bar("Eval", max=len(data.valid_x))
    for grid, x, y in zip(data.x, data.valid_x, data.valid_y):
        progress.next()
        start_time = time.time()
        x = x.reshape(-1, 1, 300, 400)
        predicted_y = model(make_gpu(x))
        predicted_y = make_numpy(predicted_y.cpu())
        predicted_y = predicted_y.reshape(300, 400)
        predicted_grid, pred_no_clusters = ndimage.label(predicted_y, structure=np.ones((3, 3)))
        end_time = time.time()

        no_true_clusters.append(extract_no_clusters(grid))
        no_predicted_clusters.append(pred_no_clusters)
        exec_times.append(end_time - start_time)
    progress.finish()

    stats("Execution time", exec_times)
    stats("Clusters no", np.array(no_true_clusters) - np.array(no_predicted_clusters))


if __name__ == "__main__":
    main()
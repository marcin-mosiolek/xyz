
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
    print("\n============ {} \n============ {} ".format(name))
    print("Mean: {}".format(np.mean(data)))
    print("Std: {}".format(np.mean(data)))
    q = [x for x in range(0, 100, 21)]
    percentiles = np.percentile(data, q=q)
    print("Percentiles:")
    for i, p in zip(q, percentiles):
        print("{}%:{:.3f}".format(i, p))


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
    for x, y in zip(data.valid_x, data.valid_y):
        progress.next()
        start_time = time.time()
        predicted_y = model(x)
        predicted_y = make_numpy(predicted_y)
        predicted_y = predicted_y.reshape(300, 400)
        predicted_grid = ndimage.label(predicted_y, structure=np.ones((3, 3)))
        end_time = time.time()

        exec_times.append(end_time - start_time)
    progress.finish()

    stats("Execution time", exect_times)


if __name__ == "__main__":
    main()
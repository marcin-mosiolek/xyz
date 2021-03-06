import torch
import numpy as np
from scipy import ndimage
import model
import tools
import time
import eval
from sklearn import metrics
from progress.bar import Bar
import sys
import matplotlib.pyplot as plt
import math

import pandas as pd

def cluster(grid):
    grid = grid.reshape(300, 400)
    grid = grid.astype(np.int32)
    return ndimage.label(grid, structure=np.ones((3, 3)))


def get_frame(data, frame_no, pivot):
    return data.x[pivot + frame_no], data.y[pivot + frame_no]


def show(frame):
    frame = frame.reshape(300, 400)
    plt.imshow(frame)


def predict(model, x):
    x = x.reshape(-1, 1, 300, 400)
    x = tools.make_gpu(x)
    return model(x)


def show_all(x, y, pred_y, labelled):
    x = x.reshape(300, 400)
    pred_y = pred_y.reshape(300, 400)
    y = y.reshape(300, 400)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    valid_true_inds = (x > 0) & (labelled > 0)
    labelled[~valid_true_inds] = 0

    ax1.imshow(x, cmap='tab20b')
    ax1.set_title("True clusters (Grid 2D)")
    ax2.imshow(labelled, cmap='tab20b')
    ax2.set_title("Predicted clusters (CCA)")

    ax3.imshow(y, cmap='tab20b')
    ax3.set_title("Expected autoencoder output (Convex hulls)")
    ax4.imshow(pred_y, cmap='gray')
    ax4.set_title("Autencoder output (Convex hulls)")

    plt.savefig("example_1.png")


def extract_labels(true_grid, predicted_grid):
    valid_inds = np.where((true_grid > 0) & (predicted_grid > 0))
    true_labels = true_grid[valid_inds]
    predicted_labels = predicted_grid[valid_inds]
    return true_labels, predicted_labels


def stats(name, data):
    print("\n============ {} ============".format(name))
    print("Mean: {}".format(np.mean(data)))
    print("Std: {}".format(np.mean(data)))
    q = [x for x in np.linspace(0, 100, 21)]
    percentiles = np.percentile(data, q=q)
    print("Percentiles:")
    for i, p in zip(q, percentiles):
        print("{}%:{:.3f}".format(i, p))


def eval_autoencdoer(autoencoder, data, frame_no, threshold=0.9, visualize=False):
    # visualize results
    x, y = get_frame(data, frame_no, data.pivot)
    x = x.reshape(300, 400)

    start_time = time.time()
    py = predict(autoencoder, data.normalize(x.copy()))

    py = tools.make_numpy(py.cpu())
    py[py >= threshold] = 255
    py[py < threshold] = 0
    labelled, _ = cluster(py)
    end_time = time.time()

    if visualize:
        show_all(x, y, py, labelled)

    true_labels, predicted_labels = extract_labels(x, labelled)
    no_true_convex_hulls = len(np.unique(x)) - 1
    no_pred_convex_hulls = len(np.unique(labelled)) - 1
    score = metrics.adjusted_mutual_info_score(true_labels, predicted_labels)

    return score, no_true_convex_hulls, no_pred_convex_hulls, end_time - start_time


def eval_baseline(autoencoder, data, frame_no, threshold=0.9, visualize=False):
    x, y = get_frame(data, frame_no, data.pivot)
    x = x.reshape(300, 400)

    # run baseline algorithm on the raw input
    closed_grid = ndimage.gray_closing(x, structure=np.ones((4, 4)))
    predicted_labels, _ = cluster(closed_grid)

    # run convex hulls algorithm
    convex_hulls = predict(autoencoder, data.normalize(x.copy()))

    # compare only the points which are present in both labellings
    convex_hulls = tools.make_numpy(convex_hulls.cpu())
    convex_hulls = convex_hulls.reshape(300, 400)
    convex_hulls[convex_hulls >= threshold] = 1
    convex_hulls[convex_hulls < threshold] = 0
    valid_inds = (x > 0) & (convex_hulls > 0)

    # extract labels
    true_labels = x[valid_inds]
    baseline_labels = predicted_labels[valid_inds]

    return metrics.adjusted_mutual_info_score(true_labels, baseline_labels)



def main():
    # create model
    autoencoder_weights = torch.load("conv_autoencoder.pth", map_location=lambda storage, loc: storage)
    autoencoder = model.AutoEncoder().cuda()
    autoencoder.load_state_dict(autoencoder_weights)

    # load data
    data = tools.DataLoader("/mnt/moria/voyage_clustering/convex_hulls2.npy", keep_original=True)
    scores = []
    baseline_scores = []
    tcs = []
    pcs = []
    ets = []

    #print(np.unique(data.x))

    progress = Bar("Evaluation: ", max=len(data.valid_x))
    for frame_no in range(0, len(data.valid_x)):
        progress.next()
        score, tc, pc, et = eval_autoencdoer(autoencoder, data, frame_no, threshold=0.9)
        baseline_score = eval_baseline(autoencoder, data, frame_no, threshold=0.9)

        if math.isnan(score):
            continue
        scores.append(score)
        tcs.append(tc)
        pcs.append(pc)
        ets.append(et)
        baseline_scores.append(baseline_score)

    progress.finish()

    df = pd.DataFrame(data={
        "Baseline" : baseline_scores,
        "Convex" : scores
    })

    df.describe(percentiles=np.linspace(0, 1, 21)).to_csv("eval.csv")

    stats("Autoencoder score", scores)
    stats("Baseline score", baseline_scores)
    stats("Convex hulls", np.abs(np.array(tcs) - np.array(pcs)))
    print("\nMean execution time: {:.4f}".format(np.mean(et)))

if __name__ == "__main__":
    main()

import torch
import numpy as np
from scipy import ndimage
import model
import tools
import time
import eval
from sklearn import metrics
from progress.bar import Bar

import matplotlib.pyplot as plt


def cluster(grid):
    grid = grid.reshape(300, 400)
    grid = grid.astype(np.int32)
    return ndimage.label(grid, structure=np.ones((3, 3)))


def get_frame(data, frame_no, pivot):
    return data.valid_x[frame_no], data.valid_y[frame_no]


def show(frame):
    frame = frame.reshape(300, 400)
    plt.imshow(frame)


def predict(model, x):
    x = x.reshape(-1, 1, 300, 400)
    x = tools.make_cpu(x)
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


def extract_no_clusters(grid):
    # -1 because it returns 0 as well
    return len(np.unique(grid) - 1)


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


def get_score_for_frame(data, frame_no, threshold=0.9, visualize=False):
    # visualize results
    x, y = get_frame(data, frame_no, 0)
    x = x.reshape(300, 400)
    py = predict(autencoder, data.normalize(x.copy()))

    py = tools.make_numpy(py)
    py[py >= threshold] = 255
    py[py < threshold] = 0
    labelled, _ = cluster(py)

    if visualize:
        show_all(x, y, py, labelled)

    true_labels, predicted_labels = extract_labels(x, labelled)
    no_true_labels = len(np.unique(true_labels))
    no_pred_labels = len(np.unique(predicted_labels))
    score = metrics.adjusted_rand_score(true_labels, predicted_labels)

    return score, no_true_labels, no_pred_labels


def main():
    # create model
    autencoder_weights = torch.load("conv_autoencoder.pth", map_location=lambda storage, loc: storage)
    autencoder = model.AutoEncoder()
    autencoder.load_state_dict(autencoder_weights)

    scores = []
    tcs = []
    pcs = []

    progress = Bar("Evaluation", max=len(data.valid_x))
    for frame_no in range(0, len(data.valid_x)):
        progress.next()
        score, tc, pc = get_score_for_frame(data, frame_no, threshold=0.9)
        scores.append(score)
        tcs.append(tc)
        pcs.append(pc)

    progress.finish()

    stats("Score", scores)
    stats("TC - PC", np,array(tcs) - np.array(pcs))

if __name__ == "__main__":
    main()
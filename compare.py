import tools
from progress.bar import Bar
from scipy import ndimage
from sklearn import metrics
import numpy as np

def stats(name, data):
    print("\n============ {} ============".format(name))
    print("Mean: {}".format(np.mean(data)))
    print("Std: {}".format(np.mean(data)))
    q = [x for x in np.linspace(0, 100, 21)]
    percentiles = np.percentile(data, q=q)
    print("Percentiles:")
    for i, p in zip(q, percentiles):
        print("{}%:{:.3f}".format(i, p))

def main():
    data = tools.DataLoader("./mnt/moria/voyage_clustering/convex_hulls2.npy", keep_original=True)

    baseline_scores = []
    convex_scores = []

    progress = Bar("Comparing: ", max=len(data.x))
    # for each frame
    for x, y in zip(data.x, data.y):
        # convert to nice size
        x = x.reshape(300, 400).astype(np.int32)
        y = y.reshape(300, 400).astype(np.int32)

        # extract pixels which are present only on the both images
        common = np.zeros_like(x)
        common[(x > 0) & (y > 0)] = 1

        # now we can run two clusterings, one based on the baseline algorithms, the other on the convex hulls
        # 1) baseline algorithm
        closed_grid = ndimage.binary_closing(common, structure=np.ones((4, 4)))
        baseline_labels, _ = ndimage.label(closed_grid, structure=np.ones((3, 3)))

        # 2) convex hulls algorithm
        norm_y = data.normalize(y.copy())
        convex_labels, _ = ndimage.label(norm_y, structure=np.ones((3, 3)))

        # extract original pixel values
        true_labels = x[(x > 0) & (y > 0)]
        baseline_labels = baseline_labels[(x > 0) & (y > 0)]
        convex_labels = convex_labels[(x > 0) & (y > 0)]

        baseline_scores.append(
            metrics.adjusted_mutual_info_score(true_labels, baseline_labels)
        )

        convex_scores.append(
            metrics.adjusted_mutual_info_score(true_labels, convex_labels)
        )

        progress.next()
    progress.finish()

    stats("Convex", convex_scores)
    stats("Baseline", baseline_scores)



if __name__ == "__main__":
    main()
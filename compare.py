import tools
from progress.bar import Bar
from scipy import ndimage
from sklearn import metrics
import math
import numpy as np
import pandas as pd
pd.set_option("precision", 3)

def main():
    data = tools.DataLoader("../autencoder/small.npy", keep_original=True)

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

        baseline_score = metrics.adjusted_mutual_info_score(true_labels, baseline_labels)
        convex_score = metrics.adjusted_mutual_info_score(true_labels, convex_labels)

        if math.isnan(baseline_score) or math.isnan(convex_score):
            continue


        baseline_scores.append(baseline_score)
        convex_scores.append(convex_score)

        progress.next()
    progress.finish()

    df = pd.DataFrame({
        "Baseline" : baseline_scores,
        "Convex" : convex_scores
    })


    df.describe(percentiles=np.linspace(0, 1, 21)).to_csv("comparison.csv")




if __name__ == "__main__":
    main()
import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage.draw import polygon
import sys

# set size boundaries
x_min_max = [-30, 50]
y_min_max = [-30, 30]
z_min_max = [-3, 5]
eps = 1e-5
cell_size = 0.2
n_cells_x = np.ceil((x_min_max[1] - x_min_max[0])/cell_size)
n_cells_y = np.ceil((y_min_max[1] - y_min_max[0])/cell_size)
print(n_cells_x, n_cells_y)

x_min_max[1] = (n_cells_x * cell_size) - np.abs(x_min_max[0])
y_min_max[1] = (n_cells_y * cell_size) - np.abs(y_min_max[0])


# load data
data = np.load("outputs/full_set.npy")
no_of_points = len(data)

# create progress bar
bar = Bar("Processed", max=no_of_points)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

dataset_X = []
dataset_Y = []

# iterate over frames
for i, frame in enumerate(data):
    bar.next()

    grid = np.zeros((int(n_cells_y), int(n_cells_x)))
    convex_hull_grid = np.zeros_like(grid)

    # for each clusters
    for cluster in frame:
        # leave only meaningful points
        valid_inds = np.where(
            (cluster[:, 0] >= x_min_max[0] + eps) &
            (cluster[:, 0] < x_min_max[1] - eps) &
            (cluster[:, 1] >= y_min_max[0] + eps) &
            (cluster[:, 1] < y_min_max[1] - eps) &
            (cluster[:, 2] >= z_min_max[0] + eps) &
            (cluster[:, 2] < z_min_max[1] - eps))[0]
        valid_points = cluster[valid_inds,:]

        # we're interested only in the aerial view
        xy_points = valid_points[:,:2]
        xy_points[:, 0] = xy_points[:, 0] - x_min_max[0]
        xy_points[:, 1] = xy_points[:, 1] - y_min_max[0]

        # convert the points into expected resolution
        cells_inds = (np.floor(xy_points / 0.2).astype(np.int64))

        # probably the most ugly piece of code I've ever written
        cells_inds = np.array(
            list(set([tuple(row) for row in cells_inds]))
        )
        if cells_inds.shape[0] < 4:
            continue
        x, y = cells_inds.swapaxes(0, 1)

        # mark the cluster on the grid
        grid[y, x] = color 

        # calculate convex hull for each cluster
        color = valid_points[0, 3]
        if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            try:
                convex_hull = ConvexHull(cells_inds)
            except:
                print(cells_inds)
                continue
            polygon_shape = cells_inds[convex_hull.vertices]

            x_hull, y_hull = polygon_shape.swapaxes(0, 1)
            rows, cols = polygon(x_hull, y_hull)
            convex_hull_grid[cols, rows] = color
        else:
            convex_hull_grid[y, x] = color

    if np.nonzero(grid)[0].shape[0]:
        dataset_X.append(grid)
        dataset_Y.append(convex_hull_grid)

    #ax1.imshow(grid)
    #ax2.imshow(convex_hull_grid)

    #plt.pause(2)
    #ax1.cla()
    #ax2.cla()

bar.finish()

print("Saving: {} frames".format(len(dataset_X)))
np.save("outputs/convex_hulls", [dataset_X, dataset_Y])

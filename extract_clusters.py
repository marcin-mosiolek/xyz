import os
import re
import pprint
import glob
import json
import pickle
import numpy as np

def cluster_select(frame, dimensions):
    x,y,z,l,w,h,angle = dimensions
    obj_frame = frame[:,:3] - np.array([x,y,z])
    s = np.sin(angle)
    c = np.cos(angle)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    obj_frame = np.matmul(obj_frame,R)
    cond1 = np.all(obj_frame >= np.array([-float(l)/2,-float(w)/2,-float(h)/2]),axis = 1)
    cond2 = np.all(obj_frame <= np.array([float(l)/2,float(w)/2,float(h)/2]),axis = 1)
    mask = np.where(cond1 & cond2)[0]
    to_draw = frame[mask,:]

    return to_draw, mask


def main():

    root_dir = '/Users/marcin/PycharmProjects/research/scale_eval_dataset_v1'
    clusters_per_frame = []

    # search all pickle files
    annotation_pickle_files = glob.glob(os.path.join(root_dir, '*.pkl'))
    pprint.pprint('pickle files found: {}'.format(annotation_pickle_files))

    # for each pickle file
    for pkl_file in annotation_pickle_files:
        data = pickle.load(
            open(os.path.join(root_dir, pkl_file), 'rb')
        )
        data_keys_sorted = sorted(data.keys(), key=lambda x: int(re.split('_|\.', x)[-2]))

        # for each json file
        for datum in data_keys_sorted:
            key = datum
            val = data[key]

            # Extract point cloud data.
            json_pcloud_filepath = os.path.join(root_dir, key)
            points_data = json.load(open(json_pcloud_filepath, 'r'))
            points_data = points_data['points']
            points = np.asarray([[p['x'], p['y'], p['z']] for i, p in enumerate(points_data)])

            # extract clusters
            clusters = []
            print("{} contains {} clusters".format(key, len(val['cuboids'])))
            for i, cuboid in enumerate(val['cuboids']):
                position = cuboid['position']
                yaw = cuboid['yaw']
                dimensions = cuboid['dimensions']
                alignment_information = (
                position['x'], position['y'], position['z'], dimensions['x'], dimensions['y'], dimensions['z'], yaw)
                cluster, mask = cluster_select(points, alignment_information)
                cluster_label = np.full(cluster.shape[0], i+1)

                # append label id
                cluster = np.column_stack([cluster, cluster_label])

                # append cluster to clusters lists
                clusters.append(cluster)
            clusters_per_frame.append(np.array(clusters))

    print("Processed {} files".format(len(clusters_per_frame)))
    np.save("extracted_clusters", clusters_per_frame)

if __name__ == "__main__":
    main()
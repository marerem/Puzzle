import numpy as np
import pandas as pd
import skimage.io
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import confusion_matrix

from segmentation import segment_pieces, extract_pieces
from feature_extraction import extract_features


def find_outlier(features, pred):
    '''
    This function aims to detect outliers in a cluster. If a cluster size minus one or two falls into [9,12,16],
    it divides the cluster into two sub-clusters.

    :param features: A 2D array where each row represents a data point and each column represents a feature of the data point.
    :param pred: A 1D array where each element represents the cluster id of a data point.

    :return mod_pred: A modified pred array where outliers have been re-clustered.
    '''
    mod_pred = pred

    cluster_id = list(np.unique(pred))
    for id in cluster_id:
        features_cluster = np.array(features)[np.squeeze(np.argwhere(pred == id)),:]
        if (features_cluster.shape[0] - 1) in [9, 12, 16] or (features_cluster.shape[0] - 2) in [9, 12, 16] :
            Z = linkage(features_cluster, method='ward')
            new_cluster = fcluster(Z, t= 2, criterion='maxclust')
            mod_pred[np.squeeze(np.argwhere(pred == id))] = np.max(cluster_id) + new_cluster


    return mod_pred


def merge_cluster(pred):
    '''
    This function merges two clusters if their combined size falls into [9,12,16].
    :param pred: A 1D array where each element represents the cluster id of a data point.
    :return mod_pred: A modified pred array where some clusters have been merged.
    '''
    mod_pred = pred

    cluster_id = list(np.unique(pred))
    len_clusters = [np.sum(pred == id) for id in cluster_id]
    for i in range(len(len_clusters)):
        if len_clusters[i] not in [9, 12, 16]:
            for j in range(i+1, len(len_clusters)):
                if len_clusters[j] not in [9, 12, 16]:
                    if (len_clusters[i] + len_clusters[j]) in [9, 12, 16] :
                        mod_pred[np.squeeze(np.argwhere(np.logical_or(pred == cluster_id[i], pred == cluster_id[j])))] = cluster_id[i]
    return mod_pred


def segment_cluster_pieces(path_img, nb_cluster, gabor_filter_bank_list, feat_idx=None, y_true=None):
    '''
    This function loads an image, segments it, extracts puzzle pieces and their features, normalizes these features,
    performs clustering, validates the clusters, and calculates the precision of the clustering.

    Parameters:
    path_img: str
        The path to the image file.
    nb_cluster: int
        The number of clusters for the clustering algorithm.
    gabor_filter_bank_list: list
        The list of Gabor filter banks.
    feat_idx: list, optional
        The indices of the relevant features to be used in clustering. If None, all features are used.
    y_true: numpy.ndarray, optional
        The true labels of the data points. If provided, the precision of the clustering will be calculated.

    Returns:
    prec: numpy.ndarray
        The precision of each cluster.
    pred: numpy.ndarray
        The predicted labels of the data points.
    puzzles: list
        The extracted puzzle pieces from the image.
    '''
    # Load image
    img = skimage.io.imread(path_img)

    # Segment image
    seg, contours = segment_pieces(img)

    # Extract puzzle pieces
    puzzles = extract_pieces(img, contours)

    # Extract features of puzzle feat
    features = [extract_features(x, gabor_filter_bank_list) for x in puzzles]
    features = np.array(pd.DataFrame(features))

    # Remplace inf + nan by median if they exist
    features[np.where(np.isinf(features))] = 0

    col_mean = np.nanmedian(features, axis=0)
    inds = np.where(np.isnan(features))
    features[inds] = np.take(col_mean, inds[1])

    # Normalize features
    features_normalized = StandardScaler().fit_transform(features)
    x_df = pd.DataFrame(features_normalized)

    # Select relevent features
    df = x_df.iloc[:, feat_idx]

    # Perform hierarchical clustering
    Z = linkage(df, method='ward')
    pred = fcluster(Z, t=nb_cluster, criterion='maxclust')

    # Check cluster have coherent size, merge + split cluster if not
    pred = find_outlier(df, pred)
    pred = merge_cluster(pred)
    pred = find_outlier(df, pred)

    if y_true != None:
        # Compute class-wise precision
        M = confusion_matrix(y_true, pred)
        prec = np.max(M, axis=1) / np.sum(M, axis=1)

        print(f'Precision for image ' + path_img + str(prec))

        return prec, pred, puzzles
    else:
        return pred, puzzles


def relabel_and_find_min_label(predicted_clusters_copy):
    '''
    This function relabels the cluster labels so they are continuous integers,
    and identifies the label with the minimum count (which we identify as an outlier).

    :param predicted_clusters_copy: List of cluster assignments for each data point.

    :return min_labels_image: List of labels that have the fewest elements in each cluster.
    :return min_label_counts: List of counts of elements associated with each label in min_labels_image.
    :return predicted_clusters_copy: List of cluster assignments for each data point, with relabeled clusters.
    '''
    min_labels_image = []
    min_label_counts = []  # stores the counts of pieces for each min_label

    for i in range(len(predicted_clusters_copy)):
        # get unique labels and their counts
        unique_labels, counts = np.unique(predicted_clusters_copy[i], return_counts=True)

        # create a mapping from old labels to new labels
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        # apply the mapping to the array
        for old_label, new_label in label_mapping.items():
            predicted_clusters_copy[i][predicted_clusters_copy[i] == old_label] = new_label

        # identify the label with the fewest elements
        unique_labels, counts = np.unique(predicted_clusters_copy[i], return_counts=True)
        min_label_index = np.argmin(counts)
        min_label = unique_labels[min_label_index]
        min_labels_image.append(min_label)

        # get the count of pieces for the min_label
        min_label_count = counts[min_label_index]
        min_label_counts.append(min_label_count)

    return min_labels_image, min_label_counts, predicted_clusters_copy

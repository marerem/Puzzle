from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.io
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from segmentation import segment_pieces, extract_pieces
from feature_extraction import extract_features

def perform_PCA_and_plot_explained_variance(features, n_components=None, exp_variance=0.95, print_results=False):
    '''
    Performs PCA on given features and plots the cumulative explained variance.
    :param features: The features to perform PCA on.
    :param n_components: The number of principal components to consider. If None, use all components.
    :param exp_variance: Find the number of principal components needed to reach exp_variance%.
    :param print_results: If True, print the explained variance per principal component and the cumulative explained variance.
    :return selected_features: The principal components selected based on the PCA.
    :return principal_components: All principal components.
    :return n_components_95: The number of principal components needed to reach 95% explained variance.
    '''
    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(features)

    # Access the principal components and explained variance ratios
    principal_components = pca.components_
    explained_variance_ratios = pca.explained_variance_ratio_

    if print_results:
        # Print the principal components and explained variance ratios
        for i, pc in enumerate(principal_components):
            print(f"Explained variation per PC {i+1}: {explained_variance_ratios[i]}")

    # Compute and print the cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratios)
    if print_results:
        print("Cumulative Explained Variance:", cumulative_variance)

    # Determine the number of principal components needed to reach exp_variance% explained variance
    n_components = np.where(cumulative_variance >= exp_variance)[0][0] + 1

    # Plot the cumulative variance to determine PC components
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Elbow Method - Cumulative Explained Variance vs. Number of PCs')
    plt.grid(True)
    plt.axvline(n_components, color='r', linestyle='--', label='{}% explained variance'.format(exp_variance*100))
    plt.legend()
    plt.show()

    selected_features = principal_components[:n_components]
    return selected_features, principal_components, n_components


def MIM(path_img, y_true, gabor_filter_bank_list):
    '''
    Extracts the features from an image, normalizes them, and uses Mutual Information (MI)
    to find the most important features.

    :param path_img: The path of the image to extract features from.
    :param y_true: The ground truth labels for the image.
    :param gabor_filter_bank_list: A list of Gabor filters used to extract features from the image.

    :return selected_features: A list of the most important features based on MI > 1. If no feature
                               passes the threshold, an empty list is returned. If only one feature passes,
                               a list with one element is returned.
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

    MI = mutual_info_classif(x_df, y_true)
    feat = np.squeeze(np.argwhere(MI > 1))

    if feat.size != 0:
        if feat.size == 1:
            return [feat]
        return list(feat)

    return []

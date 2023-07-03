import cv2
import skimage.filters
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scipy.signal
import pandas as pd

from segmentation import *
from helper_functions import *

from save_evaluation_files import save_feature_map


#%%
# COLOR FEATURES

def color_histogram(img, bins=(8,8,8)):
    '''
    Extract color histogram features from the image.
    :param img: image in RGB color space from which the features are extracted
    :param bins: number of bins for each channel
    :return hist: color histogram
    '''

    # Compute the color histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    # Normalize histogram so the sum of all bin heights is 1 and flatten
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def grey_histogram(img, bins=256):
    '''
    Extract grey level histogram features from the image.
    :param img: image in grayscale from which the features are extracted
    :param bins: number of bins for the grey level values
    :return hist: grey level histogram
    '''

    # Compute the grey level histogram
    hist = cv2.calcHist([img], [0], None, [bins], [0, 256])

    # Normalize histogram so the sum of all bin heights is 1 and flatten
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def histogram_features(histogram):
    '''
    Find the number of peaks in a histogram and the max value of the peak
    :param histogram: color or grey histogram
    :return nb_peaks: number of peaks in the histogram
    :return max_peak: max value of the peak
    '''
    # Find the peaks in the histogram
    peaks = scipy.signal.find_peaks(histogram)[0]

    # Compute number of peaks
    nb_peaks = len(peaks)

    # If there are peaks, compute the max peak value; if no peaks, max_peak = 0
    max_peak = np.max(histogram[peaks]) if nb_peaks > 0 else 0

    return nb_peaks, max_peak

def mean_std_hist(histogram):
    '''
    Compute the mean and standard deviation of the color/grey histogram, to use as features.
    :param histogram: rgb or grey color histogram
    :return mean: mean of the histogram
    :return std: standard deviation of the histogram
    '''
    mean = np.mean(histogram)
    std = np.std(histogram)
    return mean, std


def average_std_color(img):
    '''
    Compute the average and standard deviation value of the red, green and blue colors of the image.
    :param img: image in RGB color space
    :return avg_red, avg_green, avg_blue: average red, green, blue value
    :return std_red, std_green, std_blue: standard deviation red, green, blue value
    '''
    avg_red = np.mean(img[:, :, 0])
    avg_green = np.mean(img[:, :, 1])
    avg_blue = np.mean(img[:, :, 2])

    std_red = np.std(img[:, :, 0])
    std_green = np.std(img[:, :, 1])
    std_blue = np.std(img[:, :, 2])

    return avg_red, avg_green, avg_blue, std_red, std_green, std_blue

def average_std_grey(img):
    '''
    Compute the average and standard deviation value of the grey color in the image.
    :param img: image in grayscale
    :return avg_gray: average grey value
    :return std_gray: standard deviation grey value
    '''
    avg_gray = np.mean(img)
    std_gray = np.std(img)

    return avg_gray, std_gray

#%%
# TEXTURE FEATURES
def gabor_filter(ksize, sigma, theta, lambd, gamma = 1, psi = 0):
    '''
    Define a gabor filter that can be used to extract texture features.
    :param ksize: size of gabor filter, must be an odd number
    :param sigma: standard deviation of the gaussian envelope, size of image region being analyzed
    :param theta: orientation of the function, 0 is horizontal, 90 is vertical
    :param lambd: wavelength of the sinusoidal factor, frequency being looked for in the texture (high frequency = fine details/slim borders, low frequency = coarse details/thick borders)
    :param gamma: spatial aspect ratio
    :param psi: phase offset
    :return: a gabor filter
    '''
    gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    return gabor_filter

def gabor_filter_bank(ksize, sigmas, lambdas, thetas):
    '''
    Create a gabor filter bank.
    :param ksize: size of gabor filter, must be an odd number
    :param sigmas: standard deviation of the gaussian envelope, size of image region being analyzed
    :param lambdas: wavelength of the sinusoidal factor, frequency being looked for in the texture (high frequency = fine details/slim borders, low frequency = coarse details/thick borders)
    :param thetas: orientation of the function, 0 is horizontal, 90 is vertical
    :return: a gabor filter bank
    '''
    gabor_filter_bank = []
    for sigma in sigmas:
        for lambd in lambdas:
            for theta in thetas:
                gabor_filter_bank.append(gabor_filter(ksize, sigma, theta, lambd))
    return gabor_filter_bank


def apply_gabor_filter_bank(img, gabor_filter_bank_list):
    '''
    Apply a gabor filter bank to the image.
    :param img: image in RGB color space
    :param gabor_filter_bank_list: gabor filter bank (list)
    :return: list of filtered images
    '''
    filtered_images = []
    for gabor_filter in gabor_filter_bank_list:
        filtered_image = cv2.filter2D(img, cv2.CV_8UC3, gabor_filter)
        filtered_images.append(filtered_image)
    return filtered_images



def gabor_features(gabor_images):
    '''
    Extract features from gabor convoluted images, such as mean, standard deviation, kurtosis, power spectrum
    :param gabor_images: list of filtered images
    :return features_mean: list of mean values
    :return features_std: list of standard deviation values
    :return features_kurtosis: list of kurtosis values
    '''
    features_mean = []
    features_std = []
    features_kurtosis = []
    for image in gabor_images:
        # Extract mean of image as a feature
        features_mean.append(np.mean(image))
        # Extract standard deviation of image as a feature
        features_std.append(np.std(image))
        # Extract kurtosis of image as a feature
        kurtosis_value = stats.kurtosis(image.flatten(), fisher=True)
        features_kurtosis.append(kurtosis_value)

    return features_mean, features_std, features_kurtosis

# compute power spectrum of the filter responses
def power_spectrum(gabor_images):
    '''
    Compute power spectrum of the filter responses
    :param gabor_images: list of filtered images
    :return features: list of power spectrum values
    '''
    power_spectrum = []
    for image in gabor_images:
        # Apply Fourier transform
        f = np.fft.fft2(image)
        # shift the zero-frequency component to the center of the spectrum
        fshift = np.fft.fftshift(f)
        # compute the magnitude spectrum
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        pwr_spectrum = np.abs(magnitude_spectrum)**2
        power_spectrum.append(pwr_spectrum)

    return power_spectrum

def compute_power_spectrum_features(power_spectrum):
    # Flatten every power spectrum
    power_spectrum_reshape = np.array(power_spectrum).reshape((np.array(power_spectrum).shape[0], -1))

    # Compute features
    mean_power = list(np.mean(power_spectrum_reshape, axis=1))
    max_power = list(np.max(power_spectrum_reshape, axis=1))
    std_power = list(np.std(power_spectrum_reshape, axis=1))
    skewness_power = [stats.skew(x) for x in power_spectrum_reshape]
    kurtosis_power = [stats.kurtosis(x, fisher=True) for x in power_spectrum_reshape]

    return mean_power, max_power, std_power, skewness_power, kurtosis_power

#%%
# SHAPE FEATURES

def compute_shape_feat(img):
    '''
    Computes shape-based features from a given image. Specifically, it calculates the average circularity, area, and
    perimeter of all contours in the image.

    :param img: The input image from which features are to be computed.

    :return: A tuple consisting of:
             - Average circularity of all contours in the image.
             - Average area of all contours in the image.
             - Average perimeter of all contours in the image.
    '''
    # Find contours
    canny = cv2.Canny(img, 20, 50, 1)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Compute circularity of contours
    circularity, areas, perimeters = [], [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, False)
        areas.append(area)
        perimeters.append(perimeter)

        if perimeter != 0:
            circularity.append(4 * np.pi * (area / perimeter ** 2))

    return np.mean(circularity), np.mean(areas), np.mean(perimeters)
#%%
# EXTRACT COLOR AND TEXTURE FEATURES
def extract_features(img, gabor_filter_bank_list):
    '''
    Extract features from the image.
    :param img: image in RGB color space
    :param gabor_filter_bank_list: gabor filter bank (list)
    :return features: dictionary of features
    '''
    # Compute color histogram
    hist_color = color_histogram(img)

    # Compute mean and standard deviation of color histogram
    mean_color, std_color = mean_std_hist(hist_color)

    # Compute average red and green value
    avg_red, avg_green, avg_blue, std_red, std_green, std_blue = average_std_color(img)

    # convert img to greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute grey histogram
    hist_grey = grey_histogram(img)

    # Compute mean and standard deviation of grey histogram
    mean_grey, std_grey = mean_std_hist(hist_grey)

    # Find number of peaks and max peak height of grey histogram
    peaks, max_peak = histogram_features(hist_grey)

    avg_grey, std_grey = average_std_grey(img)

    # apply gabor filter bank
    gabor_images = apply_gabor_filter_bank(img, gabor_filter_bank_list)

    # compute gabor features
    mean_gabor, std_gabor, kurtosis_gabor = gabor_features(gabor_images)

    # compute power spectrum of the filter responses
    power_spectrum_ = power_spectrum(gabor_images)

    # extract features from the power spectrum
    mean_power, max_power, std_power, skewness_power, kurtosis_power = compute_power_spectrum_features(power_spectrum_)

    # extract circularity feature
    circularity, area, perimeter = compute_shape_feat(img)

    # define a dict of features
    features = {'mean_color': mean_color, 'std_color': std_color, 'avg_red': avg_red, 'avg_green': avg_green,
                'avg_blue': avg_blue,
                'mean_gabor': mean_gabor, 'std_gabor': std_gabor, 'kurtosis_gabor': kurtosis_gabor,
                'mean_power': mean_power, 'max_power': max_power, 'std_power': std_power,
                'skewness_power': skewness_power, 'kurtosis_power': kurtosis_power, 'circularity': circularity,
                'area': area, 'perimeter': perimeter, 'std_red': std_red, 'std_blue': std_blue, 'std_green': std_green,
                'mean_grey': mean_grey, 'std_grey': std_grey, 'peaks': peaks, 'max_peak': max_peak,
                'avg_grey': avg_grey, 'std_grey': std_grey}

    # Fix the format of features dict
    features_new = {}

    # Iterate over keys and values in the old dictionary
    for key, values in features.items():
        # Check if value is a list
        if isinstance(values, list):
            # Iterate over the list of values
            for i, value in enumerate(values):
                # Create a new key for each value in the list
                new_key = f"{key}_{i + 1}"
                # Add the new key-value pair to the new dictionary
                features_new[new_key] = value
        else:
            # If value is not a list, copy the key-value pair to the new dictionary
            features_new[key] = values

    return features_new


def extract_features_all_img(directory, gabor_filter_bank_list, save_features=False, saving_path=None):
    '''
    Extract features from all images in the given directory.
    :param directory: The path of the directory containing the images.
    :param gabor_filter_bank_list: gabor filter bank (list)
    :param save_features: If True, save the features
    :param saving_path: The path of the directory where the features will be saved.
    :return all_features: A list where each element is a numpy array containing normalized features of each puzzle piece from each image in the directory.
    '''
    all_features = []
    if save_features:
        os.makedirs(saving_path, exist_ok=True)

    for i, file_name in enumerate(sorted(os.listdir(directory))):

        if file_name.endswith('.png'):
            # Load image
            img = skimage.io.imread(directory + '/' + file_name)

            # Segment image
            seg, contours = segment_pieces(img)

            # Extract puzzle pieces
            puzzles = extract_pieces(img, contours)

            # Extract features of puzzle feat
            features = [extract_features(x, gabor_filter_bank_list) for x in puzzles]
            features = np.array(pd.DataFrame(features))

            # Replace inf + nan by median if they exist
            features[np.where(np.isinf(features))] = 9999

            col_mean = np.nanmedian(features, axis=0)
            inds = np.where(np.isnan(features))
            features[inds] = np.take(col_mean, inds[1])

            # Normalize features
            features_normalized = StandardScaler().fit_transform(features)

            all_features.append(features_normalized)

            # Save feature map if save_features is True
            if save_features:
                save_feature_map(i, features_normalized, saving_path)

    return all_features

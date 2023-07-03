## load images
import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import math



def save_segmented_image(image_index, seg, folder ="segments", path="data_project"):
    '''
    Save the segmented image into a folder
    :param image_index: index of the image to save
    :param seg: numpy array of the segmented image
    :param folder: folder where the segmented image will be stored
    :param path: path to the data
    '''
    path_segments = os.path.join(path, folder)
    if not os.path.isdir(path_segments):
        os.makedirs(path_segments)

    print(path_segments)
    filename = os.path.join(path_segments, "segmented_{}.png".format(str(image_index).zfill(2)))
    Image.fromarray(seg).save(filename)

def save_extracted_pieces(image_index, pieces, folder ="extracts", path="data_project"):
    '''
    Save the extracted puzzle pieces into a folder
    :param image_index: index of the image from which the pieces were extracted
    :param pieces: list of numpy arrays of the extracted pieces
    :param folder: folder where the extracted pieces will be stored
    :param path: path to the data
    '''
    path_pieces = os.path.join(path, folder, "extract_{}".format(str(image_index).zfill(2)))
    if not os.path.isdir(path_pieces):
        os.makedirs(path_pieces)

    print(path_pieces)
    for i, piece in enumerate(pieces):
        filename = os.path.join(path_pieces, "piece_{}.png".format(str(i).zfill(2)))
        Image.fromarray(piece).save(filename)


def load_input_image(image_index, folder="train2", path="data_project"):
    '''
    Load input image from the dataset
    :param image_index: index of the image to load
    :param folder: folder where the image is stored
    :param path: path to the data
    :return: input image as a numpy array
    '''

    filename = "train_{}.png".format(str(image_index).zfill(2))
    path_solution = os.path.join(path, folder, filename)

    im = Image.open(os.path.join(path, folder, filename)).convert('RGB')
    im = np.array(im)
    return im


def save_solution_puzzles(image_index, solved_puzzles, outliers, folder="train2", path="data_project", group_id=32):
    '''
    Save solved puzzles and outliers into a folder
    :param image_index: index of the image to save
    :param solved_puzzles: list of numpy arrays containing the solved puzzles
    :param outliers: list of numpy arrays containing the outliers
    :param folder: folder where the image is stored
    :param path: path to the data
    :param group_id: group id
    '''
    path_solution = os.path.join(path, folder + "_solution_{}".format(str(group_id).zfill(2)))
    if not os.path.isdir(path_solution):
        os.mkdir(path_solution)

    print(path_solution)
    for i, puzzle in enumerate(solved_puzzles):
        filename = os.path.join(path_solution, "solution_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(puzzle).save(filename)

    for i, outlier in enumerate(outliers):
        filename = os.path.join(path_solution, "outlier_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(outlier).save(filename)


def solve_and_export_puzzles_image(image_index, folder="train2", path="data_project", group_id="00"):
    """
    Wrapper function to load image and save solution
    :param image_index: index of the image to load (index number of the dataset)
    :param folder: folder where the image is stored
    :param path: path to the data
    :param group_id: group id
    """

    # open the image
    image_loaded = load_input_image(image_index, folder=folder, path=path)
    # print(image_loaded)

    ## call functions to solve image_loaded
    solved_puzzles = [(np.random.rand(512, 512, 3) * 255).astype(np.uint8) for i in range(2)]
    outlier_images = [(np.random.rand(128, 128, 3) * 255).astype(np.uint8) for i in range(3)]

    save_solution_puzzles(image_index, solved_puzzles, outlier_images, folder=folder, group_id=group_id)

    return image_loaded, solved_puzzles, outlier_images

def display_images_in_grid(puzzle_pieces, image_index = None, cmap = None, folder='grids', path='data_project', save=False):
    '''
    Display the puzzle pieces in a grid.
    :param puzzle_pieces: list of puzzle pieces
    :param image_index: index of the image
    :param cmap: color map
    :param folder: folder where the grid is saved
    :param path: path to the folder
    :param save: boolean that indicates if the grid is saved
    '''

    # Determine grid size based on the number of images
    grid_size = math.ceil(math.sqrt(len(puzzle_pieces)))

    fig, axs = plt.subplots(grid_size, grid_size)

    for i, ax in enumerate(axs.flatten()):
        if i < len(puzzle_pieces):
            if cmap is None:
                ax.imshow(puzzle_pieces[i])
            else:
                ax.imshow(puzzle_pieces[i], cmap=cmap)
            ax.axis('off')  # Hide axes
        else:
            fig.delaxes(ax)  # Remove empty subplots

    fig.suptitle('Number of puzzle pieces: ' + str(len(puzzle_pieces)), fontsize=12)

    # Create the directory if it does not exist
    path_grid = os.path.join(path, folder)
    if not os.path.isdir(path_grid):
        os.makedirs(path_grid)

    if save:
        # Save the grid to a file
        filename = os.path.join(path_grid, "grid_{}.png".format(str(image_index).zfill(2)))
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_color_histogram(img):
    '''
    Plot the color histogram of an image, as well as the image itself.
    param img: image of which the color histogram is plotted
    '''
    # Compute color histograms
    red_hist = np.histogram(img[:,:,0], bins=256, range=[0,256])
    green_hist = np.histogram(img[:,:,1], bins=256, range=[0,256])
    blue_hist = np.histogram(img[:,:,2], bins=256, range=[0,256])

    # Plot histograms
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 4, 1)
    plt.bar(red_hist[1][:-1], red_hist[0], color='red')
    plt.title('Red Histogram')
    plt.subplot(1, 4, 2)
    plt.bar(green_hist[1][:-1], green_hist[0], color='green')
    plt.title('Green Histogram')
    plt.subplot(1, 4, 3)
    plt.bar(blue_hist[1][:-1], blue_hist[0], color='blue')
    plt.title('Blue Histogram')
    plt.subplot(1, 4, 4)
    plt.imshow(img)
    plt.show()


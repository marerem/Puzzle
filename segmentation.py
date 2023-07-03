import cv2
import numpy as np

# SEGMENTATION FUNCTIONS:
def segment_pieces(img, save_mask=False):
    '''
    Segment the puzzle pieces.
    :param img: img to be segmented
    :return seg: img with drawn segmentation lines
    :return save_mask: if True, save the mask of the segmentation
    :return seg, contours: if save_mask is False, return the segmentation and the contours of the segmentation
    :return seg, mask, contours: if save_mask is True, return the segmentation, the mask and the contours of the segmentation
    '''

    img_copy = img.copy()
    mask = np.zeros_like(img)

    # Preprocessing: blur image
    blur = cv2.medianBlur(img, ksize=21)

    # Find edges
    canny = cv2.Canny(blur, 20, 50, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(canny, kernel, iterations=3)

    # Fill in big contours to remove edges inside pieces
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    fill_in = np.zeros_like(img)
    for c in contours:
        fill_in = cv2.drawContours(fill_in, [c], 0, (255, 0, 0), thickness=cv2.FILLED)

    # Find minrect fitting in fill_in contours
    canny = cv2.Canny(fill_in, 10, 100, 1)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        seg = cv2.drawContours(img_copy, [box], 0, (0, 255, 0), 3)
        if save_mask:
            cv2.drawContours(mask, [box], 0, (255), -1)

    if save_mask:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.where(mask > 0, 1, 0)
        return seg, mask, contours
    else:
        return seg, contours


def extract_pieces(img, contours):
    '''
    Extract the puzzle pieces using the segmentation.
    :param img: image from which the pieces are extracted
    :param contours: contours of puzzle pieces on the image
    :return pieces: list of the extracted puzzle pieces
    '''

    # empty list to store the extracted pieces
    pieces = []
    for c in contours:
        # Find angle and center of min rectangle fitting in each contours
        rect = cv2.minAreaRect(c)
        angle, center = rect[2], rect[0]
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Extract only current puzzle piece on image
        temp = cv2.drawContours(np.zeros_like(img), [box], 0, (255, 255, 255), thickness=cv2.FILLED)
        idx_puzzle = np.where(temp == 255)
        piece = np.zeros_like(img);
        piece[idx_puzzle] = img[idx_puzzle]

        # Rotate image so that piece is centered
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_piece = cv2.warpAffine(piece, rot_mat, piece.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Extract piece by cropping image
        idx_piece = np.where(rotated_piece != 0)
        x, y = np.min(idx_piece[1]) + 10, np.min(idx_piece[0]) + 10
        crop = rotated_piece[y:y + 128, x:x + 128, :]
        pieces.append(crop)

    return pieces

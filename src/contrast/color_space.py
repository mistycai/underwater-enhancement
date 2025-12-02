# src/contrast/color_space.py
import cv2
import numpy as np


def bgr_to_lab_luminance(bgr_img):
    '''
    Convert a BGR image to Lab and return the L (luminance) channel
    '''
    # BGR -> Lab
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L = lab_img[:, :, 0].astype(np.float32)
    L_norm = L / 255.0

    return L_norm, lab_img


def lab_luminance_to_bgr(L_norm, lab_img):
    '''
    Replace the L channel in a Lab image with a new luminance map
    (float32 in [0, 1]) and convert back to BGR
    '''
    L_uint8 = np.clip(L_norm * 255.0, 0, 255).astype(np.uint8)
    # replace L channel
    lab_copy = lab_img.copy()
    lab_copy[:, :, 0] = L_uint8
    # Lab -> BGR
    bgr_out = cv2.cvtColor(lab_copy, cv2.COLOR_LAB2BGR)
    return bgr_out
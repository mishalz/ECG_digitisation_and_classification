from pdf2image import convert_from_path
import glob,os
import os, subprocess
import math
from typing import Tuple, Union
import cv2
import numpy as np
from deskew import determine_skew
from matplotlib import pyplot as plt
import uuid

output_jpeg_path = "/content/drive/MyDrive/ECG images/ECG1.jpg"
path = '/content/drive/MyDrive/ECG images'
filename = 'Pre-processed Image'
pdf_image = '/content/drive/MyDrive/ECG images/ECG7.pdf'

def convert_pdf_to_jpeg(input_pdf_path, output_jpeg_path):
    # Convert the PDF to a list of JPEG images
    images = convert_from_path(input_pdf_path)

    # Save the first image as a JPEG file
    images[0].save(output_jpeg_path, 'JPEG')


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def image_preprocessing(pdf_file):
    # function to convert pdf file to jpeg

    # Path to the input PDF file
    input_pdf_path = pdf_file

    # Convert PDF to JPEG and save it
    convert_pdf_to_jpeg(input_pdf_path, output_jpeg_path)

    # reads the jpeg file
    image = cv2.imread(output_jpeg_path)

    # deskew function
    angle = determine_skew(image)
    rotated_img = rotate(image, angle, (0, 0, 0))

    # split the image into three channels
    red, green, blue = cv2.split(rotated_img)

    # image smoothing using Gaussian blur
    blur = cv2.GaussianBlur(blue, (625, 625), 1)

    # de noise the image through adaptive thresholding
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 121, 25)

    # Invert the binary image
    binary = cv2.bitwise_not(th)

    # define kernel for morphological operator
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    # Apply morphological closing operation to fill the gaps of a signal
    closed_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # return the pre processed image
    return closed_image
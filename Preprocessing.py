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

#output_jpeg_path = "/content/drive/MyDrive/ECG images/ECG1.jpg"
#path = '/content/drive/MyDrive/ECG images'
filename = 'Pre-processed Image'
#pdf_image = '/content/drive/MyDrive/ECG images/ECG7.pdf'

#def convert_pdf_to_jpeg(input_jpeg_path , output_jpeg_path):
    # Convert the PDF to a list of JPEG images
    #images = convert_from_path(input_jpeg_path)
    # Save the first image as a JPEG file
    #images[0].save(output_jpeg_path, 'JPEG')


def deskew_image(image, ang):
  # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

  # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

  # Calculate the rotated bounding rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]

  # Rotate the image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, ang, 1.0)
    deskewed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
  
    return deskewed_image


def image_preprocessing(image):
    image = np.array(image)
    # function to convert pdf file to jpeg
    
    # Path to the input PDF file
    #input_pdf_path = pdf_file

    # Convert PDF to JPEG and save it
    #convert_pdf_to_jpeg(input_pdf_path, output_jpeg_path)

    # reads the jpeg file
    #image = cv2.imread(image)

    # deskew function
     # Deskew the image
    #angle = determine_skew(image)
    rotated_img = deskew_image(image, 0.2) #for ecg 54 angle = 1.5 #for ecg 7 angle=0.2
    

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
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import rank_filter
import matplotlib.pyplot as plt
import csv
from scipy.signal import resample

total_boxes_width = 62
time_per_box = 0.04

total_boxes_height = 10
voltage_per_box = 0.1

total_voltage = total_boxes_height * voltage_per_box
total_time = total_boxes_width * time_per_box 

voltage_h = 0
# voltage_h = ecg_height

x_offset = 0
y_offset = voltage_h

def getIsoElectricLines(image, projecting_axis, peak_distance, apply_kernel, doMax, peak_height=None):
    '''
    ----- Definition -----
    This method calculates a projection of the input image into a specified axis which is then used to find
    the segmenting lines.

    ----- Parameters -----
    1. image: the input image
    2. projecting_axis (int) : this axis in which to get the image's projection (0 for vertical and 1 for horizontal)
    3. peak_distance (double) : this is a hyperparameter on which the finding peak function is tuned.
                                It is the minimum distance that should be present between two peaks.
    4. apply_kernel (bool) : tells the function if a guassian kernel should be applied onto the projection histogram.
    5. doMax (bool) : determines if the a max function should be applied to convert the projection into 1D
    6. peak_height (double) : this is a hyperparameter on which the finding peak function is tuned.
                              It is the minimum height that every peak that is to be detected should have.

    ----- Returns -----
    an array of peaks that are detected within the specified constraints (through the hyperparameters)
    '''

    # finding the projection in a specified axis
    hist = np.sum(image, axis=projecting_axis)

    # if the resulting projection is 2D or more
    if (doMax):
        hist = hist.max(axis=1)

    # kernel filter configurations
    kernel_size = 51
    sigma = 20

    # applying the guassian kernel to smoothen the curves of the projection histogram
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    if (apply_kernel):
        smoothed_hist = cv2.filter2D(hist, -1, kernel).flatten()
        smoothed_hist = smoothed_hist * -1
        hist = smoothed_hist

    # returns the peaks with the specified hyperparameters
    peaks, _ = find_peaks(hist, height=peak_height, distance=peak_distance)

    return peaks


def drawIsoElectricLines(image, peaks, axis):
    '''
    ----- Definition -----
    Draws the segmenting lines that are calculated using the getIsoElectricLines function.
    For visually analysing how the image will be segmented.

    ----- Parameters -----
    1. image: the input image on which to draw the segmenting lines
    2. peaks: (array) an array of peaks in the specified axis
    3. axis: (int) determines the axis of the segmenting lines (0 for vertical and 1 for horizontal)

    ----- Returns -----
    None

    '''
    imgwithlines = np.copy(image)

    for point in peaks:
        if (axis == 0):
            startingPos = (point, 0)
            endingPos = (point, imgwithlines.shape[0])
        else:
            startingPos = (0, point)
            endingPos = (imgwithlines.shape[1], point)

        cv2.line(imgwithlines, startingPos, endingPos, (0, 0, 255), 2)
        
    # Display image using matplotlib
    plt.imshow(cv2.cvtColor(imgwithlines, cv2.COLOR_BGR2RGB))
    plt.show()


def getContours(image, kernel_size):
    '''
    ----- Definition -----
    For finding all the contours in an image

    ----- Parameters -----
    1. image: the input image of which the contours are to be found
    2. kernel_size (int) : length of the kernel filter

    ----- Returns -----
    contours: array of all the contours detected in the image

    '''
    # edge detection
    qrs = cv2.Canny(image, 150, 200, L2gradient=True)

    # create a structuring element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # dilate the edges
    dilated_edges = cv2.dilate(qrs, kernel, iterations=1)

    # Apply erosion to the image
    # img_eroded = cv2.erode(dilated_edges, kernel, iterations = 1)
    # closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, None)

    # apply thresholding to extract the ECG signal
    ret, thresh = cv2.threshold(dilated_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # RETR_EXTERNAL flag is more efficient and faster as it only returns the external contours.
    # RETR_TREE flag retrieves all of the contours (both internal and external) and reconstructs a full hierarchy of nested contours.
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    return contours, largest_contour

def getSignalFromContour(image, startingline):

    # get the largest contour (which should be the ECG signal)
    _, largest_contour = getContours(image,1)
    # compute the bounding box of the ECG contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # extract that part from the original image
    ecg_signal = image[startingline: x + h, x:x+1000]
    # finding the ECG contour from the bounded region
    _, ECG_contour = getContours(ecg_signal,1)
    return ECG_contour

def drawECGSignal(points):

  x_values, y_values = zip(*points)

  plt.figure(figsize=(8.9, 1.3))
  plt.plot(x_values, y_values)

  plt.xlabel('Time (s)')
  plt.ylabel('Voltage (mV)')
  plt.show()

def getScaledECG(total_time, width, total_voltage, voltage_height, x_offset, y_offset, contour):

  '''
  ----- Definition -----
  calibrates all the points in an array with a scale that is calculated from the specified parameters 
  and then sorts the points in ascending order

  ----- Parameters -----
  1. total_time: the total time that the ECG image or contour covers
  2. width: width in pixels of that contour
  3. total_voltage: the total voltage that is represented within the bounding rectangle
  4. voltage_height: the height in pixels of the area which represents the total voltage
  5. x_offset: value for calibrating the points to start from origin
  6. y_offset: value for calibrating the points to start from origin
  7. contour: The points on which to apply calibration

  ----- Returns -----
  List of the calibrated and sorted points
  
  '''

  x_scale =  total_time / width
  y_scale = total_voltage / voltage_height

  points = []
  for point in contour:
      x, y = point[0]
      x = round((x - x_offset) * x_scale , 4)
      y = round((y_offset - y) * y_scale , 4)
      points.append((x, y))
  
  drawECGSignal(points)

  return sorted(points, key=lambda x: x[0])

def getResampledECG(points, resampling_frequency):
  
  '''
  ----- Definition -----
  applies a new sampling frequency to the input list of points

  ----- Parameters -----
  1. points: (list of tuples where each tuple contains a time and corresponding voltage value)
  2. resampling_frequency: the new sampling frequency

  ----- Returns -----
  returns a list of tuples which the resampled values

  '''
  time, voltage = zip(*points)

  time = np.array(time)
  voltage = np.array(voltage)
  
  current_frequency = len(time) / time[-1]
  print(current_frequency)

  num_samples = int(len(time) * (resampling_frequency / current_frequency))

  time_resampled = np.linspace(time[0], time[-1], num=num_samples)
  voltage_resampled = resample(voltage, num_samples)

  #converting the resample time and voltage values into the format of a list of tuples (t,v).
  resampled_ecg_signal = list(zip(time_resampled, voltage_resampled))
  
  return resampled_ecg_signal

def writeECGtoCSV(filename, ecg_points):
  
  with open(f'{filename}.csv','w',newline ='') as file:
    writer = csv.writer(file)
    writer.writerow(["time","MLII"])
    for row in ecg_points:
      writer.writerow(row)
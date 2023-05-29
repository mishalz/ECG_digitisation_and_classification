from email.mime import image
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import rank_filter
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.signal import resample
from scipy import signal
from scipy.ndimage import interpolation
from Preprocessing import image_preprocessing
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.figure import Figure


def getIsoElectricLines(image, projecting_axis, peak_distance,apply_kernel, doMax,peak_height = None):
 
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
  hist = np.sum(image,axis = projecting_axis)
  
  # if the resulting projection is 2D or more
  if(doMax):
    hist = hist.max(axis=1)

  # kernel filter configurations
  kernel_size = 51
  sigma = 20

  # applying the guassian kernel to smoothen the curves of the projection histogram
  kernel = cv2.getGaussianKernel(kernel_size, sigma)
  if(apply_kernel):
    smoothed_hist = cv2.filter2D(hist, -1, kernel).flatten()
    smoothed_hist = smoothed_hist*-1
    hist = smoothed_hist

  # returns the peaks with the specified hyperparameters
  peaks, _ = find_peaks(hist, height = peak_height , distance= peak_distance)

  return peaks

def getContours(image, kernel_size):

  '''
  ----- Definition -----
  Applies Edge Detection to find all the contours in an image

  ----- Parameters -----
  1. image: the input image of which the contours are to be found
  2. kernel_size (int) : length of the kernel filter

  ----- Returns -----
  contours: array of all the contours detected in the image
  largest_contour: the largest contour amongst all the detected contours
  
  '''
  # edge detection
  qrs = cv2.Canny(image, 150, 200, L2gradient=True)

  # create a structuring element
  kernel = np.ones((kernel_size,kernel_size), np.uint8)

  # dilate the edges
  dilated_edges = cv2.dilate(qrs, kernel, iterations=1)

  # apply thresholding to extract the ECG signal
  ret, thresh = cv2.threshold(dilated_edges, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  #RETR_EXTERNAL flag is more efficient and faster as it only returns the external contours. 
  #RETR_TREE flag retrieves all of the contours (both internal and external) and reconstructs a full hierarchy of nested contours.
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  largest_contour = max(contours, key=cv2.contourArea)
  
  # Initialize variables
  longest_contour = None
  max_arc_length = 0

  # Iterate over the contours
  for contour in contours:
      arc_length = cv2.arcLength(contour, True)
      if arc_length > max_arc_length:
          max_arc_length = arc_length
          longest_contour = contour

  return contours, longest_contour

  
def drawECGSignal(points):


  '''
  ----- Definition -----
  For plotting a ECG signal

  ----- Parameters -----
  1. points: The array of (time, voltage) coordinates of the signal

  ----- Returns -----
  none
  
  '''
  x_values, y_values = zip(*points)

  '''plt.figure(figsize=(8.9, 1.3))
  plt.plot(x_values, y_values)

  plt.xlabel('Time (s)')
  plt.ylabel('Voltage (mV)')
  plt.show()'''
  fig = Figure(figsize=(8.9, 1.3))
  ax = fig.add_subplot(111)
  ax.plot(x_values, y_values)

  ax.set_xlabel('Time (s)')
  ax.set_ylabel('Voltage (mV)')
   # Adjust the margins within the subplot
  ax.margins(x=0.05, y=0.05)

  fig.tight_layout()  # Ensure all elements fit within the figure

  fig.savefig('ecg_plot.png')

def getScaledECG(contour, x_offset, y_offset, x_scale, y_scale):

  '''
  ----- Definition -----
  calibrates all the points in an array with a scale that is calculated from the specified parameters 
  and then sorts the points in ascending order

  ----- Parameters -----
  1. contour: The points on which to apply calibration
  2. x_offset: value for calibrating the points in the x axis to start from origin
  3. y_offset: value for calibrating the points in the y axis to start from origin
  4. x_scale: the value of time per pixel
  5. y_scale: the value of voltage per pixel 

  ----- Returns -----
  List of the calibrated and sorted points
  
  '''
  points = []
  xpoints = []

  for point in contour:
      x, y = point[0]
 
      #To check if the voltage value corresponding to this time value already exists
      if np.invert(np.isin(x, xpoints)):
        scaled_x = (x - x_offset) * x_scale 
        scaled_y = (y_offset - y) * y_scale 
        points.append((scaled_x, scaled_y))
        xpoints.append(x)

  
 
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

  num_samples = int(len(time) * (resampling_frequency / current_frequency))

  time_resampled = np.linspace(time[0], time[-1], num=num_samples)
  voltage_resampled = resample(voltage, num_samples)

  #converting the resample time and voltage values into the format of a list of tuples (t,v).
  resampled_ecg_signal = list(zip(time_resampled, voltage_resampled))
  
  return resampled_ecg_signal

def getBeats( voltage, peakIndexes):

  '''
  ----- Definition -----
  finds the peaks in a given ecg signal and segments them according to 90 pixels before the peak 
  and 144 after the peaks, since the window size has to be 234

  ----- Parameters -----
  1. voltage: array of the voltage values at the current frequency
  2. resampling_frequency: the new sampling frequency

  ----- Returns -----
  returns an a list of np.arrays each of which contain a specific beat

  '''

  beats = []

  for peakIndex in peakIndexes:
    startingIndex = int(peakIndex - (0.25*360)) 
    endingIndex = int(peakIndex + (0.40*360))

    if (startingIndex > 0) and ( endingIndex < len(voltage)):
      beats.append(np.array(voltage[startingIndex: endingIndex]))
  return beats

def writeECGtoCSV(filename, ecg_points):
  
  '''
  ----- Definition -----
  writes the ecg_points to a CSV file

  ----- Parameters -----
  1. filename: The filename in which to store the points.
  2. ecg_points: The ECG values that are to be stored.

  ----- Returns -----
  none

  '''

  with open(filename,'w',newline ='') as file:
    writer = csv.writer(file)
    headings = []
    headings.append('sample id')
    headings.append('record')
    headings.append('beat')
    headings.append('label')
    for index in range(234):
      headings.append(f'x-{index}')

    writer.writerow(headings)
    for row in ecg_points:
      writer.writerow(row)

def getDigitisedSignal(image1):
  image=image_preprocessing(image1)
  '''
  ----- Definition -----
  Includes all the necessary functions to digitise the ecg 
  (segmenting Lead 2 -> Finding the baseLine for Lead 2 -> Finding the region of interest 
  -> Extracting the ECG signal from the region of interest -> scaling the points)

  ----- Parameters -----
  1. filename: The filename in which to store the points.
  2. ecg_points: The ECG values that are to be stored.

  ----- Returns -----
  The scaled digitised ECG points 

  '''
  
  # Loading the ECG record image using the provided record name
  img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


  # Cropping the ECG image into two different versions for more accurate segmentation
  # so that we can only focus on the lead-II region
  cropped_img = img[: , : int(img.shape[1]*0.50) ]
  cropped_img_vertical = img[:,: int(img.shape[1]*0.3)]


  #---------- Horizontal Segmenting Lines ---------
  vert_peaks = getIsoElectricLines(cropped_img_vertical, 0, 200, False,True) #(80000,120000) for ecg 7, distance=1 for ecg 54 and 200 for ecg7
  
  # extracting the values for only the first and last peak needed for segmenting the ECG vertically
  first_peak = vert_peaks[0]
  highest_peak = vert_peaks[-1]

  vert_peaks = [first_peak, highest_peak]


  #-------------- Horizontal Segmenting Lines -------------
  horizontal_peaks = getIsoElectricLines(cropped_img, 1, 120, True, True)
  
  #-------------- Segmenting Lead II -------------
  grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #Segmenting the lead 2 based on the horizontal and vertical isoelectric lines
  cropped_lead_2 = grayscale[horizontal_peaks[-2]:horizontal_peaks[-1],  vert_peaks[0]:vert_peaks[1]]
  
  #-------------- Baseline for our lead-II wave ------------
  lead_2_peaks = getIsoElectricLines(cropped_lead_2,1, 100,False,False) #150 for ecg7, 100 for ecg 54

  # this is the height of the rectangular scale indicator
  ecg_height= lead_2_peaks[1] - lead_2_peaks[0]

  #-------------- Signal Extraction -------------

  # Find all contours and get the largest contour (which should be the ECG signal)
  _, largest_contour = getContours(cropped_lead_2,2)
  
  # compute the bounding box of the ECG contour
  x, y, w, h = cv2.boundingRect(largest_contour)

  # extract that part from the original image
  ecg_signal = cropped_lead_2[lead_2_peaks[0]: x + h, x:x+1000]

  # finding the ECG contour from the bounded region
  _, ECG_contour = getContours(ecg_signal,1)


  #--------------- Scaling the ECG -------------

  #scaling parameters
  total_boxes_width = 62
  time_per_box = 0.04

  total_boxes_height = 10
  voltage_per_box = 0.1

  sampling_frequency = 360

  total_voltage = total_boxes_height * voltage_per_box
  total_time = total_boxes_width * time_per_box 

  total_number_of_samples = int(total_time * sampling_frequency)
  voltage_h = ecg_height

  x_offset = 0
  y_offset = voltage_h

  x_scale =  total_time / w
  y_scale = total_voltage / voltage_h

  # returns us the scaled ecg points 
  sorted_points =  getScaledECG(ECG_contour, x_offset, y_offset, x_scale, y_scale)


  return sorted_points


def downloadDigitisedSignal(scaled_points, className, record_name):
  '''
  ----- Definition -----
  Resamples the points to 360 HZ, segments into beats and then store them into a specified format into csv

  ----- Parameters -----
  1. filename: The filename in which to store the points.
  2. ecg_points: The ECG values that are to be stored.

  ----- Returns -----
  none

  '''
  resampled_points = getResampledECG(scaled_points, 360)

  # finding the peaks in the data 
  time, voltage = zip(*resampled_points)

  time = np.array(time)
  voltage = np.array(voltage)

  ECGpeaks, _ = find_peaks(voltage, distance = 200)

  beats = getBeats(voltage, ECGpeaks)
  ecg = []
  for index in range(len(beats)):
    ecg.append((index, record_name, index+1, className, *beats[index] ))
   
    # Ask the user to choose the file path and name
  filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
  if filename:
    writeECGtoCSV(filename, ecg)
      
    messagebox.showinfo("Download", "CSV file has been downloaded successfully.")

  
  print('written to csv')



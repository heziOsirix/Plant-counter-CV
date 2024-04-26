# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import global_functions
import os
TB_WIN_LABEL = "Trackbars"

MIN_HUE_TB_LABEL = "min Hue"
MIN_SAT_TB_LABEL = "min Sat"
MIN_VAL_TB_LABEL = "min Val"

MAX_HUE_TB_LABEL = "max Hue"
MAX_SAT_TB_LABEL = "max Sat"
MAX_VAL_TB_LABEL = "max Val"

MIN_HUE_VALUE = 0
MIN_SAT_VALUE = 0
MIN_VAL_VALUE = 0

MAX_HUE_VALUE = 180
MAX_SAT_VALUE = 255
MAX_VAL_VALUE = 255

# declaring min hue value for green
MIN_GREEN_HUE = 45
MAX_GREEN_HUE = 77
MIN_GREEN_SAT = 19
MAX_GREEN_SAT = 255
MIN_GREEN_VAL = 164
MAX_GREEN_VAL = 255

KERNEL_SIZE_TB_LABEL = "kernel size"
DEFAULT_KERNEL_SIZE = 2
MAX_KERNEL_SIZE = 10

ERODE_ITERATIONS_TB_LABEL = "erode"
DEFAULT_ERODE_ITERATIONS = 2
MAX_ERODE_ITERATIONS = 10

DILATE_ITERATIONS_TB_LABEL = "dilate"
DEFAULT_DILATE_ITERATIONS = 4
MAX_DILATE_ITERATIONS = 30

ESCAPE_KEY = 27
RESIZABLE_WINDOW = 0

def nothing(x):
  pass
def create_masked_image(image):
  # small = cv2.resize(src, (0, 0), fx=SMALL_FACTOR, fy=SMALL_FACTOR)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                        # convert to gray scale
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                          # convert to HSV
  kernelSize = 2
  dilateIterations = 4
  erodeIterations = 2
  lower = np.array([0, 0, 0])
  upper = np.array([255, 255, 255])
  mask = cv2.inRange(hsv, lower, upper)
  kernel = np.ones((kernelSize, kernelSize), np.uint8)

    
  masked = cv2.bitwise_and(gray, gray, mask=mask)
  masked = cv2.threshold(masked, 5, 255, cv2.THRESH_BINARY)[1]
  masked = cv2.erode(masked, kernel, iterations = erodeIterations)
  masked = cv2.dilate(masked, kernel, iterations = dilateIterations)
  return masked
  
def main(srcPath, dstPath):
  SMALL_FACTOR = 0.3 
  show_images = False
  src = cv2.imread(srcPath)     # read the source file
  small = cv2.resize(src, (src.shape[1],src.shape[0]), fx=SMALL_FACTOR, fy=SMALL_FACTOR)      # ressize the image  to new dim

  
  if dstPath != None:
    global_functions.ensureDir(dstPath)

  masked = create_masked_image(small)

  #CHANGE TO MASKED IN ORDER TO WORK ON THE ORIGINAL IMAGE
  if show_images:
    cv2.imshow('masked image',masked)
    cv2.waitKey()
    cv2.destroyAllWindows()

  #CHANGE TO MASKED IN ORDER TO WORK ON THE ORIGINAL IMAGE
  contours, _ = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contoursLen = len(contours)
  plantsNumber = 0
  colorStep = int(200.0/contoursLen)
  PERIMETER_LIMIT = 10
  LINE_WIDTH = 1
  
  for i in range(contoursLen):
    perimeter = cv2.arcLength(contours[i], True)
    if perimeter > PERIMETER_LIMIT and perimeter < 70:
        plantsNumber += 1
        val = (i+1) * colorStep
        cv2.drawContours(src, [contours[i]], -1, (0,0,255), LINE_WIDTH)
        #print("(" + str(val) + "," + str(val) + "," + str(val) + ") : " + str(perimeter))
  
  print("\n" + str(plantsNumber) + " plants.")
  if show_images:
    cv2.imshow("Contours", src)

    cv2.waitKey()
    cv2.destroyAllWindows()
  if dstPath != None:
    directory, filename_with_extension = os.path.split(srcPath)
    filename, extension = os.path.splitext(filename_with_extension)
    cv2.imwrite(f"{dstPath}\\{filename}-marked.jpg", src)
    
  cv2.destroyAllWindows()
  return plantsNumber, src

def printUsage():
  print("""
  USAGE:
  python Plant_counter.py --src <img-path> [--dst <img-path>]
  e.g.: 
        
  python Plant_counter.py --src foo/bar.jpg
  """)

def parseArgs(args):
  src, dst = None, None
  
  for i in range(len(args)):
    try:
      if args[i] == "--src":
        src = args[i+1]
      elif args[i] == "--dst":
        dst = args[i+1]
    except:
      break
  
  if src == None:
    printUsage()
    sys.exit()
    
  return src, dst
 
if __name__ == "__main__":
  main(R"C:\Users\Hezid\Documents\GitHub\plant-estimation\test\OTSU_R_piece_1x8.jpg", R"C:\Users\Hezid\Documents\GitHub\plant-estimation\test")
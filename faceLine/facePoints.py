import cv2
import numpy as np

# This below mehtod will draw all those points which are from 0 to 67 on face one by one.
def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False):
  points = []
  for i in range(startpoint, endpoint+1):
    point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
    points.append(point)

  points = np.array(points, dtype=np.int32)
  cv2.polylines(image, [points], isClosed, (0, 0, 0), thickness=10, lineType=cv2.LINE_8)
#FILL
def drawPoints_Fill(image, faceLandmarks, startpoint, endpoint):
  points = []
  for i in range(startpoint, endpoint+1):
    point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
    points.append(point)

  points = np.array(points, dtype=np.int32)
  cv2.fillPoly(image, [points], (0, 0, 0), lineType=cv2.LINE_AA)

# Use this function for 70-points facial landmark detector model
# we are checking if points are exactly equal to 68, then we draw all those points on face one by one
def facePoints(image, faceLandmarks):
    assert(faceLandmarks.num_parts == 68)
    drawPoints(image, faceLandmarks, 0, 16)           # Jaw line
    drawPoints(image, faceLandmarks, 17, 21)          # Left eyebrow
    drawPoints(image, faceLandmarks, 22, 26)          # Right eyebrow
    drawPoints(image, faceLandmarks, 27, 30)          # Nose bridge
    drawPoints_Fill(image, faceLandmarks, 30, 35)    # Lower nose
    drawPoints_Fill(image, faceLandmarks, 36, 41)    # Left eye
    drawPoints_Fill(image, faceLandmarks, 42, 47)    # Right Eye
    drawPoints_Fill(image, faceLandmarks, 48, 59)    # Outer lip
    drawPoints_Fill(image, faceLandmarks, 60, 67)    # Inner lip
    # drawPoints(image, faceLandmarks, 21, 22, True) 
    # drawPoints(image, faceLandmarks, 39, 42, True)    
   
# Use this function for any model other than
# 70 points facial_landmark detector model
def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
  for p in faceLandmarks.parts():
    cv2.circle(image, (p.x, p.y), radius, color, -1)

def ycrb_mask(image):
  face_img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
  #cv2.imwrite('./img/' + str(picture_number) + '_2.jpg', image)
  lower = np.array([30,133,77], dtype = np.uint8)
  upper = np.array([255,173,127], dtype = np.uint8)
  skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
  skin = cv2.bitwise_and(image, image, mask = skin_msk)

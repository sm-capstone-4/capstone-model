import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))


#-- 데이터 파일과 이미지 파일 경로
predictor_file = '../컴퓨터비전/SMU-2022-2-Computer-Vision_Code/03_videos_and_cameras2/shape_predictor_68_face_landmarks.dat' #-- 자신의 개발 환경에 맞게 변경할 것
image_file = './b.jpg' #-- 자신의 개발 환경에 맞게 변경할 것

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
print("Number of faces detected: {}".format(len(rects)))


for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[ALL]
    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
        cv2.putText(image, "{}".format(i + 1), (x, y - 2),
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.rectangle(image, pt1=(x, y), pt2=(x+1, y+1),
                  color=(0, 255, 0), thickness=2)

face_img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

lower = np.array([0,133,77], dtype = np.uint8)
upper = np.array([255,173,127], dtype = np.uint8)
skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
skin = cv2.bitwise_and(image, image, mask = skin_msk)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)
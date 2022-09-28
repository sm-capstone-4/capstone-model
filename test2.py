import numpy as np
import dlib
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

predictor_file = '../컴퓨터비전/SMU-2022-2-Computer-Vision_Code/03_videos_and_cameras2/shape_predictor_68_face_landmarks.dat' #-- 자신의 개발 환경에 맞게 변경할 것
image_file = './f.jpg' #-- 자신의 개발 환경에 맞게 변경할 것

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Face Landmark", image)

FaceDetector = dlib.get_frontal_face_detector()
faces = FaceDetector(image)
for face in faces:
      x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
      cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2),
                    color=(0, 255, 0), thickness=2)
      img = image[y1:y2, x1:x2]

cv2.imshow("Face Landmark", img)
face_img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

lower = np.array([30,133,77], dtype = np.uint8)
upper = np.array([255,173,127], dtype = np.uint8)
skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
skin = cv2.bitwise_and(img, img, mask = skin_msk)


image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3)) # height, width 통합
print(image.shape)
k = 3 # 예제는 5개로 나누겠습니다
clt = KMeans(n_clusters = k)
clt.fit(image)

for center in clt.cluster_centers_:
    print(center)

hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

cv2.imshow("Face Landmark", skin)
# cv2.imshow("Face Landmark", img)
cv2.waitKey(0)
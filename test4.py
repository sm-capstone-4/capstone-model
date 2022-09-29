import numpy as np
import dlib
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976
import colorsys
import csv
import pandas as pd



def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab


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


f = open("test_2.csv", "w", newline="")
wr = csv.writer(f)
predictor_file = '../컴퓨터비전/SMU-2022-2-Computer-Vision_Code/03_videos_and_cameras2/shape_predictor_68_face_landmarks.dat' #-- 자신의 개발 환경에 맞게 변경할 것

picture_number_list, rgb_extraction, rgb_code_list, lab_code, skin_tone_list, hsv_list, vbs_list, real_rgb_code_list, vbs_skin_tone, rgb_overlap, black_rgb = [], [], [], [], [], [], [], [], [], [], []
person_rgb, person_rgb_r, person_rgb_g, person_rgb_b, person_rgb_list = [], [], [], [], []


for i in range(0,300):

    try:
        picture_number = i
        picture_number_list.append(picture_number)
        image_file = 'kaggle/1 ('+ str(picture_number) +').jpg' #-- 자신의 개발 환경에 맞게 변경할 것

        image = cv2.imread(image_file)
        # cv2.imshow("Face Landmark", image)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Face Landmark", gray)
        # cv2.waitKey(0)

        FaceDetector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        faces = FaceDetector(gray)


        for face in faces:
              x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
              cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2),
                            color=(0, 255, 0), thickness=2)
              img = image[y1:y2, x1:x2]

        # cv2.imshow("Face Landmark", img)
        # cv2.waitKey(0)

        cv2.imwrite('./img/' + str(picture_number) + '_1.jpg', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("Face Landmark", gray)
        # cv2.waitKey(0)

        faces = FaceDetector(gray)
        h, w, c = img.shape
        # cv2.imshow("Face Landmark", img)
        # cv2.waitKey(0)
        circle_size = (h + w) // 40

        for face in faces:
              shape = predictor(gray, face)
              for n in range(0, 68):
                  x = shape.part(n).x
                  y = shape.part(n).y
                  cv2.circle(img, (x, y), circle_size, (0, 0, 0), -1)

        # cv2.imshow("Face Landmark", img)
        # cv2.waitKey(0)

        face_img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cv2.imwrite('./img/' + str(picture_number) + '_2.jpg', img)
        lower = np.array([30,133,77], dtype = np.uint8)
        upper = np.array([255,173,127], dtype = np.uint8)
        skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
        skin = cv2.bitwise_and(img, img, mask = skin_msk)


        # cv2.imshow("Face Landmark", skin)
        # cv2.waitKey(0)

        cv2.imwrite('./img/' + str(picture_number) + '_3.jpg', skin)
        image = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
        # cv2.imshow("rgb로 바꿈", image)
        # cv2.waitKey(0)
        cv2.imwrite('./img/' + str(picture_number) + '_4.jpg', image)
        image = image.reshape((image.shape[0] * image.shape[1], 3)) # height, width 통합



        k = 2
        clt = KMeans(n_clusters = k)
        clt.fit(image)

        rgb_list = clt.cluster_centers_

        last_value = None
        overlap_count = 0
        # flag = 0
        count = 0
        for center in clt.cluster_centers_:
            print(center)
            count += 1
            person_rgb.append(center)
            rgb = person_rgb[-1]
            if count > 1:
                picture_number_list.append(picture_number)
            wr.writerow([picture_number, rgb[0], rgb[1], rgb[2]])
            print(picture_number)
            print(rgb[0])
            print(rgb[1])
            print(rgb[2])
        if count == 0:
            del picture_number_list[-1]


        print(picture_number_list)
        print(person_rgb)
    except:
        print("앙 오류띠")
        del picture_number_list[-1]
    print(len(picture_number_list))
    print(len(person_rgb))





for color_code in person_rgb:
    person_rgb_list.append(color_code)

for color_code in person_rgb_list:
    if color_code is None:
        person_rgb_r.append(None)
        person_rgb_g.append(None)
        person_rgb_b.append(None)
    else:
        person_rgb_r.append(color_code[0])
        person_rgb_g.append(color_code[1])
        person_rgb_b.append(color_code[2])



df = pd.DataFrame({
    "Picture_number" : picture_number_list,
    "rgb_r" : person_rgb_r,
    "rgb_g" : person_rgb_g,
    "rgb_b" : person_rgb_b,

})



print(df)
df.to_csv('./test.csv', sep=',', na_rep='NaN')



print(rgb_list)
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

cv2.imshow("Face Landmark", skin)
# cv2.imshow("Face Landmark", img)
cv2.waitKey(0)
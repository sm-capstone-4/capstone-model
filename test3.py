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

picture_number_list, rgb_extraction, rgb_code_list, lab_code, skin_tone_list, hsv_list, vbs_list, real_rgb_code_list, vbs_skin_tone, rgb_overlap = [], [], [], [], [], [], [], [], [], []

for i in range(0,10):


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
    flag = 0
    for center in clt.cluster_centers_:
        print(center)
        # if 30 <= int(center[0]) <= 255 and 133 <= int(center[1]) <= 173 and 77 <= int(center[2]) <= 127:
        if int(center[0]) + int(center[1]) + int(center[2]) >= 50:
            last_value = center
            rgb_code_list.append(center)
            print(center)
            overlap_count += 1
            flag = 1
            print("rgb" + str(overlap_count) + " 들어갔다")
    # print(len(rgb_code_list))
    if overlap_count == 1:
        rgb_overlap.append(None)
    else:
        rgb_overlap.append(overlap_count)
        for _ in range(overlap_count-1):
            del rgb_code_list[-1]

    # print(rgb_code_list)
    # print(last_value)
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    if flag == 0:
        print("사람 피부 범워 rgb 못 찾음")
        rgb_code_list.append(None)
        skin_tone_list.append(None)
        vbs_skin_tone.append(None)
        real_rgb_code_list.append(None)
        lab_code.append(None)
        hsv_list.append(None)
        vbs_list.append(None)


    else:
        lab_code.append(rgb2lab ( rgb_code_list[picture_number] ))
        # print(lab_code)

        spring = [65.29, 13.71, 22.31]
        summer = [66.11, 13.99, 19.33]
        autumn = [60.02, 16.61, 26.18]
        winder = [65.10, 14.55, 18.40]
        spring = LabColor(lab_l=65.29, lab_a=13.71, lab_b=22.31)
        summer = LabColor(lab_l=66.11, lab_a=13.99, lab_b=19.33)
        autumn = LabColor(lab_l=60.02, lab_a=16.61, lab_b=26.18)
        winter = LabColor(lab_l=65.10, lab_a=14.55, lab_b=18.40)



        color_delta_e = []
        skin_rgb = LabColor(lab_l=lab_code[picture_number][0], lab_a=lab_code[picture_number][1], lab_b=lab_code[picture_number][2])
        color_delta_e.append(delta_e_cie1976(skin_rgb, spring))
        color_delta_e.append(delta_e_cie1976(skin_rgb, summer))
        color_delta_e.append(delta_e_cie1976(skin_rgb, autumn))
        color_delta_e.append(delta_e_cie1976(skin_rgb, winter))
        # print(color_delta_e)
        your_tone = color_delta_e.index(min(color_delta_e))
        if your_tone == 0:
            print("너는 봄 톤")
            skin_tone_list.append("Spring")
        elif your_tone == 1:
            print("너는 여름 톤")
            skin_tone_list.append("Summer")
        elif your_tone == 2:
            print("너는 가을 톤")
            skin_tone_list.append("Autumn")
        elif your_tone == 3:
            print("너는 겨울 톤")
            skin_tone_list.append("Winter")

        rgb = rgb_code_list[picture_number]
        # print(rgb)
        real_rgb_code_list.append(rgb)
        hsv_list.append(colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255))

        vbs_list.append([lab_code[picture_number][0], lab_code[picture_number][2], hsv_list[picture_number][1]])
        # print("picture_number : ", picture_number)
        # print("rgb :", rgb_code_list)
        # print("lab :", lab_code)
        # print("hsv :", hsv_list)
        # print("vbs :", vbs_list)
        # print("lab_skin_tone :", skin_tone_list)

        vbs_zero_point = [65.20, 18, 50, 0.33]
        if lab_code[picture_number][0] > vbs_zero_point[0] and lab_code[picture_number][2] > vbs_zero_point[1] and \
                hsv_list[picture_number][1] > vbs_zero_point[2]:
            vbs_skin_tone.append("Spring warm bright")
        elif lab_code[picture_number][0] > vbs_zero_point[0] and lab_code[picture_number][2] > vbs_zero_point[1] and \
                hsv_list[picture_number][1] < vbs_zero_point[2]:
            vbs_skin_tone.append("Spring warm light")
        elif lab_code[picture_number][0] > vbs_zero_point[0] and lab_code[picture_number][2] < vbs_zero_point[1] and \
                hsv_list[picture_number][1] < vbs_zero_point[2]:
            vbs_skin_tone.append("Summer cool light")
        elif lab_code[picture_number][0] < vbs_zero_point[0] and lab_code[picture_number][2] < vbs_zero_point[1] and \
                hsv_list[picture_number][1] < vbs_zero_point[2]:
            vbs_skin_tone.append("Summer cool mute")
        elif lab_code[picture_number][0] < vbs_zero_point[0] and lab_code[picture_number][2] > vbs_zero_point[1] and \
                hsv_list[picture_number][1] < vbs_zero_point[2]:
            vbs_skin_tone.append("Autumn warm mute")
        elif lab_code[picture_number][0] < vbs_zero_point[0] and lab_code[picture_number][2] > vbs_zero_point[1] and \
                hsv_list[picture_number][1] > vbs_zero_point[2]:
            vbs_skin_tone.append("Autumn warm deep")
        elif lab_code[picture_number][0] < vbs_zero_point[0] and lab_code[picture_number][2] < vbs_zero_point[1] and \
                hsv_list[picture_number][1] > vbs_zero_point[2]:
            vbs_skin_tone.append("Winter cool deep")
        elif lab_code[picture_number][0] > vbs_zero_point[0] and lab_code[picture_number][2] < vbs_zero_point[1] and \
                hsv_list[picture_number][1] > vbs_zero_point[2]:
            vbs_skin_tone.append("Winter cool bright")
        else:
            vbs_skin_tone.append("None")
        # print(len(skin_tone_list))
        # print(len(vbs_skin_tone))

        wr.writerow([picture_number, real_rgb_code_list[picture_number][0], real_rgb_code_list[picture_number][1], real_rgb_code_list[picture_number][2],
                     lab_code[picture_number][0], lab_code[picture_number][1], lab_code[picture_number][2], hsv_list[picture_number][0], hsv_list[picture_number][1], hsv_list[picture_number][0],
                     lab_code[picture_number][0], lab_code[picture_number][2], hsv_list[picture_number][1], skin_tone_list[picture_number], vbs_skin_tone[picture_number]])

        print(len(picture_number_list))
        print(len(real_rgb_code_list))
        print(len(lab_code))
        print(len(hsv_list))
        print(len(skin_tone_list))
        print(len(vbs_skin_tone))

rgb_r_list, rgb_g_list, rgb_b_list, lab_l_list, lab_a_list, lab_b_list, hsv_h_list, hsv_s_list, hsv_v_list = [],[],[],[],[],[],[],[],[]

for color_code in real_rgb_code_list:
    if color_code is None:
        rgb_r_list.append(None)
        rgb_g_list.append(None)
        rgb_b_list.append(None)
    else:
        rgb_r_list.append(color_code[0])
        rgb_g_list.append(color_code[1])
        rgb_b_list.append(color_code[2])

for color_code in lab_code:
    if color_code is None:
        lab_l_list.append(None)
        lab_a_list.append(None)
        lab_b_list.append(None)
    else:
        lab_l_list.append(color_code[0])
        lab_a_list.append(color_code[1])
        lab_b_list.append(color_code[2])

for color_code in hsv_list:
    if color_code is None:
        hsv_h_list.append(None)
        hsv_s_list.append(None)
        hsv_v_list.append(None)
    else:
        hsv_h_list.append(color_code[0])
        hsv_s_list.append(color_code[1])
        hsv_v_list.append(color_code[2])


print(len(rgb_r_list))
print(len(rgb_g_list))
print(len(rgb_b_list))
print(len(lab_l_list))
print(len(lab_a_list))
print(len(lab_b_list))
print(len(hsv_h_list))
print(len(hsv_s_list))
print(len(hsv_v_list))
print(len(skin_tone_list))
print(len(vbs_skin_tone))


df = pd.DataFrame({
    "Picture_number" : picture_number_list,
    "rgb_r" : rgb_r_list,
    "rgb_g" : rgb_g_list,
    "rgb_b" : rgb_b_list,
    "lab_l" : lab_l_list,
    "lab_a" : lab_a_list,
    "lab_b" : lab_b_list,
    "hsv_h" : hsv_h_list,
    "hsv_s" : hsv_s_list,
    "hsv_v" : hsv_v_list,
    "vbs_v" : lab_l_list,
    "vbs_b" : lab_b_list,
    "vbs_s" : hsv_s_list,
    "skin_tone" : skin_tone_list,
    "vbs_skin_tone" : vbs_skin_tone
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
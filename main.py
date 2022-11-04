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
import pymysql



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
# predictor_file = '../컴퓨터비전/SMU-2022-2-Computer-Vision_Code/03_videos_and_cameras2/shape_predictor_68_face_landmarks.dat' #-- 자신의 개발 환경에 맞게 변경할 것

picture_number_list, rgb_extraction, rgb_code_list, lab_code, skin_tone_list, hsv_list, vbs_list, real_rgb_code_list, vbs_skin_tone, rgb_overlap = [], [], [], [], [], [], [], [], [], []


for i in range(0,1):

    picture_number = i
    picture_number_list.append(picture_number)
    # image_file = 'kaggle/1 ('+ str(picture_number) +').jpg' #-- 자신의 개발 환경에 맞게 변경할 것
    image_file = './myeong_su.jpg'
    # 이미지 읽어오기
    image = cv2.imread(image_file)

    # 읽어온 이미지 show
    # cv2.imshow("read iamge", image)
    # cv2.waitKey(0)


    # 읽어온 image 색체계 BGR -> Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray로 바꾼 이미지 show
    # cv2.imshow("Bgr -> Gray", gray)
    # cv2.waitKey(0)

    # 면상 인식 모델 가져오기
    FaceDetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    # 가져온 모델에 gray 색체계로 만들 image 삽입
    faces = FaceDetector(gray)

    # 얼굴 인식
    for face in faces:
        # 얼굴 x1, x2 ,y1, y2 함수 통해서 가져옴
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # 가져온 4개의 자표로 얼굴 사각형 그리기
        cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2),
                      color=(0, 255, 0), thickness=2)

        # 사각형으로 그린부분 잘라서 img 변수에 저장
        img = image[y1:y2, x1:x2]

    # 사각형으로 자른 얼굴 show
    # cv2.imshow("face recognization and save", img)
    # cv2.waitKey(0)

    # img 폴터에 이미지이름_1.jpg 형식으로 저장
    cv2.imwrite('./img/' + str(picture_number) + '_1.jpg', img)

    # 얼굴 인식해서 잘라 저장한 img를 BGR -> Gray 로 색체계 변경
    # gray 변수에 저장
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray 저장한거 show
    # cv2.imshow("Face Landmark", gray)
    # cv2.waitKey(0)

    # 다시 얼굴 넣기
    faces = FaceDetector(gray)

    # height, width, channel 변수에 저장
    h, w, c = img.shape

    # height, width, channel 크기에 따라
    # 적당한 크기의 써클을 그리기 위해
    # 여러 수치로 테스트 해보다가 40이 적당해서
    # 40으로 나눔
    # 그래서 circle size를 설정함
    circle_size = (h + w) // 30

    # 이제 68개의 점을 그릴건데
    # 위에서 구한 circle_size로 그릴거임
    for face in faces:
          shape = predictor(gray, face)
          for n in range(0, 68):
              x = shape.part(n).x
              y = shape.part(n).y
              cv2.circle(img, (x, y), circle_size, (0, 0, 0), -1)

    # 짠 다 그려서 한번 보자
    # cv2.imshow("draw circle", img)
    # cv2.waitKey(0)

    # 그리고 이미지번호_2.jpg로 저장해줌
    cv2.imwrite('./img/' + str(picture_number) + '_2.jpg', img)

    # YCrCb 색체계의 사람 피부색 범위를 알기 때문에
    # 일단 위에서 원까지 다그린 img 이미지를 YCrCb로 바꿔줌
    # 그걸 face_img_ycrcb에 저장
    face_img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 그래서 ycrcb 사람 피부색 범위가 (00, 133, 77) ~ (255, 173, 127)이 범위래
    # 근데 첫번째 00~255은 그냥 30 ~ 255로 내가 바꿈
    # 왜냐 이게 더 잘되는 듯?
    # 그리고 논문에도 133 ~ 173 이랑 77 ~ 127 내용은 봤는데
    # 00 ~ 255는 못본듯? 다시 봐야될 듯
    # 뭐 아무튼 이렇게 최솟값 lower랑 최댓값 upper를 저장함
    lower = np.array([30,133,77], dtype = np.uint8)
    upper = np.array([255,173,127], dtype = np.uint8)

    # 그리고 위에 face_img_ycrcb 저장한거에
    # inRange 함수써서 mask 엎어버리고
    skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)


    # 이걸 기존 이미지에 마스크 해줌
    skin = cv2.bitwise_and(img, img, mask = skin_msk)

    # 그리고 보면 이렇게 보일 거임
    # 피부색 범위가 아닌 요소는 검은색으로 덮어 버린거
    # cv2.imshow("skin_msk", skin)
    # cv2.waitKey(0)

    # 그리고 저장해줌 이름_3.jpg 확인해보셈
    cv2.imwrite('./img/' + str(picture_number) + '_3.jpg', skin)

    # 그리고 이제 RGB 값을 추출할거라]
    # BGR -> RGB로 바꿔줌
    image = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)

    # 그리고 저장하면 바뀐게 보일거임
    cv2.imwrite('./img/' + str(picture_number) + '_4.jpg', image)

    # reshape
    image = image.reshape((image.shape[0] * image.shape[1], 3)) # height, width 통합

    # k-means 색추출
    k = 2
    clt = KMeans(n_clusters = k)
    clt.fit(image)

    # clt.cluster_centers_
    # 여기에 추출한 rgb코드들이 array 형태로 저장 됨
    # array[[r,g,b], [r,g,b]] 이런식으로
    rgb_list = clt.cluster_centers_


    last_value = None
    overlap_count = 0
    flag = 0

    # 그래서 clt.cluster_centers_ 반복 돌아서
    # center에 첫 튠에 [r, g, b ] 다음 튠 [r, g, b ]  이렇게 들어갈거임
    for center in clt.cluster_centers_:
        # print(center)
        # if 30 <= int(center[0]) <= 255 and 133 <= int(center[1]) <= 173 and 77 <= int(center[2]) <= 127:

        # 여기서 center[0] = r / center[1] = g / center[2] = b 가 됨.
        # 그래서 검은색이 대체로 3개 다 더해도 숫자가 낮아서 70이하로 설정해 놓음
        # 그래서 그냥 rgb 추출해서
        # rgb_code_list 여기에 [ [r, g, b ]] 형태로 저장해 줌
        # 94 61 40
        if int(center[0]) >= 94 and int(center[1]) >= 61 and int(center[2]) >= 40:
            last_value = center
            rgb_code_list.append(center)
            # print(center)
            overlap_count += 1
            flag = 1
            # print("rgb" + str(overlap_count) + " 들어갔다")
        # else:
        #     black_rgb.append(center)
    # print(len(rgb_code_list))
    if overlap_count == 1:
        rgb_overlap.append(None)
    else:
        rgb_overlap.append(overlap_count)
        for _ in range(overlap_count-1):
            del rgb_code_list[-1]

    # 여기는 추출한 rgb matplot으로 보여줌
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    plt.figure()
    plt.axis("off")
    # plt.imshow(bar)
    # plt.show()

    # 만약에 rgb 코드 추출 하나도 못하면
    if flag == 0:
        print("사람 피부 범워 rgb 못 찾음")
        rgb_code_list.append(None)
        skin_tone_list.append(None)
        vbs_skin_tone.append(None)
        real_rgb_code_list.append(None)
        lab_code.append(None)
        hsv_list.append(None)
        vbs_list.append(None)
        # black_rgb.append(None)

    # rgb 코드 추출 하면
    else:
        # rgb코드를 lab으로 바꿔주는 함수 통해서
        # lab_code에 변환한 값 넣어 줌
        lab_code.append(rgb2lab ( rgb_code_list[picture_number] ))

        # 이거는 논문 참고해서 계절별 lab 코드
        spring = [65.29, 13.71, 22.31]
        summer = [66.11, 13.99, 19.33]
        autumn = [60.02, 16.61, 26.18]
        winter = [65.10, 14.55, 18.40]

        # 계절별 lab코드
        # Delta E 구하기 전 사전작업
        spring = LabColor(lab_l=65.29, lab_a=13.71, lab_b=22.31)
        summer = LabColor(lab_l=66.11, lab_a=13.99, lab_b=19.33)
        autumn = LabColor(lab_l=60.02, lab_a=16.61, lab_b=26.18)
        winter = LabColor(lab_l=65.10, lab_a=14.55, lab_b=18.40)


        # 사용자 lab이랑 계절별 lab이랑의 delta E를 구해서 저장할 리스트 선언
        color_delta_e = []

        # 똑같이 사용자 lab코드 Delta E 구하기 전 사전작업
        skin_rgb = LabColor(lab_l=lab_code[picture_number][0], lab_a=lab_code[picture_number][1], lab_b=lab_code[picture_number][2])
        print('labcode=', lab_code[picture_number][0], lab_code[picture_number][1], lab_code[picture_number][2])

        # 거리 구해라 뿅
        color_delta_e.append(delta_e_cie1976(skin_rgb, spring))
        color_delta_e.append(delta_e_cie1976(skin_rgb, summer))
        color_delta_e.append(delta_e_cie1976(skin_rgb, autumn))
        color_delta_e.append(delta_e_cie1976(skin_rgb, winter))

        print(color_delta_e)
        # your_tone 변수에 거리 제일 짧은거 인덱스 넣어줘
        your_tone = color_delta_e.index(min(color_delta_e))


        # 그래서 가장 짧은 인덱스가 뭥지 확인해서
        # 어떤 계절의 톤인지 skin_tone_list에 저장
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


        # rgb_code_list가 array로 저장되어 있어서 한번 뺴줘야 됨
        # 해당 picture_number꺼 array 빼서 list형태로 rgb에 저장
        rgb = rgb_code_list[picture_number]

        # list 형태로 뺸 rgb를 real_rgb_code_list에 저장
        real_rgb_code_list.append(rgb)

        # 함수 이용해서 rgb -> hsv 로 변환
        hsv_list.append(colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255))

        # vbs 저장
        # v : Lab의 L / b : Lab의 b / s : hsv의 s
        vbs_list.append([lab_code[picture_number][0], lab_code[picture_number][2], hsv_list[picture_number][1]])

        # 이건 논문에서 50몇명을 대상으로 뽑은 중앙값
        # vbs_zero_point
        vbs_zero_point = [65.1587, 17.6091, 0.3487762]

        # 논문에 있는 분류 방법
        if lab_code[picture_number][0] > vbs_zero_point[0] and lab_code[picture_number][2] > vbs_zero_point[1] and \
                hsv_list[picture_number][1] > vbs_zero_point[2]:
            vbs_skin_tone.append("따뜻한 봄 톤")
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



        # test_2.csv에 실시간 작성 코드
        wr.writerow([picture_number, real_rgb_code_list[picture_number][0], real_rgb_code_list[picture_number][1], real_rgb_code_list[picture_number][2],
                     lab_code[picture_number][0], lab_code[picture_number][1], lab_code[picture_number][2], hsv_list[picture_number][0], hsv_list[picture_number][1], hsv_list[picture_number][0],
                     lab_code[picture_number][0], lab_code[picture_number][2], hsv_list[picture_number][1], skin_tone_list[picture_number], vbs_skin_tone[picture_number]])

        # 리스트 별 길이 확인
        # print(len(picture_number_list))
        # print(len(real_rgb_code_list))
        # print(len(lab_code))
        # print(len(hsv_list))
        # print(len(skin_tone_list))
        # print(len(vbs_skin_tone))
        # print(len(black_rgb))

# 지금 rgb, lab, hsv 전부 다 (00, 00 ,00) 이렇게 저장되어 있어서 하나씩 빼줘야 됨
rgb_r_list, rgb_g_list, rgb_b_list, lab_l_list, lab_a_list, lab_b_list, hsv_h_list, hsv_s_list, hsv_v_list, black_r, black_g, black_b= [],[],[],[],[],[],[],[],[], [], [] ,[]

# # black은 걍 무시
# # for color_code in black_rgb:
# #     if color_code is None:
# #         black_r.append(None)
# #         black_g.append(None)
# #         black_b.append(None)
# #     else:
# #         black_r.append(color_code)
# #         black_g.append(color_code)
# #         black_b.append(color_code)

# rgb -> r, g, b 추출
for color_code in real_rgb_code_list:
    if color_code is None:
        rgb_r_list.append(None)
        rgb_g_list.append(None)
        rgb_b_list.append(None)
    else:
        rgb_r_list.append(color_code[0])
        rgb_g_list.append(color_code[1])
        rgb_b_list.append(color_code[2])

# lab -> l, a, b 추출
for color_code in lab_code:
    if color_code is None:
        lab_l_list.append(None)
        lab_a_list.append(None)
        lab_b_list.append(None)
    else:
        lab_l_list.append(color_code[0])
        lab_a_list.append(color_code[1])
        lab_b_list.append(color_code[2])

# hsv -> h, s, v 추출
for color_code in hsv_list:
    if color_code is None:
        hsv_h_list.append(None)
        hsv_s_list.append(None)
        hsv_v_list.append(None)
    else:
        hsv_h_list.append(color_code[0])
        hsv_s_list.append(color_code[1])
        hsv_v_list.append(color_code[2])


# print(len(rgb_r_list))
# print(len(rgb_g_list))
# print(len(rgb_b_list))
# print(len(lab_l_list))
# print(len(lab_a_list))
# print(len(lab_b_list))
# print(len(hsv_h_list))
# print(len(hsv_s_list))
# print(len(hsv_v_list))
# print(len(skin_tone_list))
# print(len(vbs_skin_tone))


# test.csv 에 저장
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
    "vbs_skin_tone" : vbs_skin_tone,
})

print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()

# print(df)
# df.to_csv('./test.csv', sep=',', na_rep='NaN')

# print("your tone is " + vbs_skin_tone[0])
# print("1st | ", "라끌랑 | ", "슈퍼쉴드 옴므쿠션 | ", "2호 | ", "39,000 | ", "delta_e : 5.767586131129729")
# print("2st | ", "라끌랑 | ", "슈퍼쉴드 옴므쿠션 | ", "3호 | ", "39,000 | ", "delta_e : 5.94556202389648")
# print("3st | ", "랑콤 | ", "똉 이돌 롱라스팅 파운데이션 | ", "P0-02 | ", "79,000 | ", "delta_e : 7.3462673365458135")
# print("4st | ", "랑콤 | ", "똉 이돌 롱라스팅 파운데이션 | ", "BO-O3 | ", "79,000 | ", "delta_e : 11.52433762868825")
# print("5st | ", "헤라 | ", "실키 스테이 24H 롱웨이 파운데이션 | ", "27N1 | ", "68,000 | ", "delta_e : 13.378566207931256")

# DB 연결하기
db = pymysql.connect(host="127.0.0.1", user="root", password="1234", db="condb", charset="utf8")

# DB 커서 만들기
cursor = db.cursor(pymysql.cursors.DictCursor)
sql = "SELECT * FROM condb.cosmetics_rgb;"
cursor.execute(sql)
table = []
result = cursor.fetchall()
for record in result:
    table.append(list(record.values()))
db.close()

print(table)
print(len(table))
delta_e_list = []
cosmetics_lab_list = []
for i in range(len(table)):
    cosmetics_lab_list.append(LabColor(lab_l=table[i][5], lab_a=table[i][5], lab_b=table[i][5]))
    delta_e_list.append(delta_e_cie1976(skin_rgb, cosmetics_lab_list[i]))
    table[i].append(delta_e_cie1976(skin_rgb, cosmetics_lab_list[i]))

table.sort(key=lambda x:x[8])
del table[5:]
for i in table:
    print(i)


# 5, 6, 7


# print(rgb_list)
# hist = centroid_histogram(clt)
# bar = plot_colors(hist, clt.cluster_centers_)
# plt.figure()
# plt.axis("off")
# plt.imshow(bar)
# plt.show()
#
# cv2.imshow("Face Landmark", skin)
# # cv2.imshow("Face Landmark", img)
# cv2.waitKey(0)
import cv2
import numpy as np

#Open a simple image
img=cv2.imread("e.jpg")
back=cv2.imread("XMY-136-B.png")
#converting from gbr to hsv color space
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#skin color range for hsv color space 
HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#converting from gbr to YCbCr color space
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#skin color range for hsv color space 
YCrCb_mask = cv2.inRange(img_YCrCb, (30, 133, 77),(255, 173, 127)) 
YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

#merge skin detection (YCbCr and hsv)
global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
global_mask=cv2.medianBlur(global_mask,3)
global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


HSV_result = cv2.bitwise_not(HSV_mask)
YCrCb_result = cv2.bitwise_not(YCrCb_mask)
global_result=cv2.bitwise_not(global_mask)
mask_res = cv2.bitwise_and(img, img, mask = HSV_mask)

#back_mask skin color
img1 = np.zeros((480, 640, 3), dtype=np.uint8) 
h,w,c = img.shape
skin_tone = np.full((h,w,c),(203,218,243),dtype=np.uint8)

mask_res2 = cv2.bitwise_and(skin_tone, skin_tone, mask = YCrCb_mask)

mask_res3 = cv2.bitwise_and(mask_res2, img, mask = YCrCb_result)

mask_res4 = cv2.bitwise_and(mask_res3, img, mask = YCrCb_result)
# 241,218,203
#show results
cv2.imshow("1_HSV.jpg",YCrCb_result)
# cv2.imshow("2_YCbCr.jpg",YCrCb_result)
# cv2.imshow("3_global_result.jpg",global_result) 
cv2.imshow("sa.jpg",skin_tone)
# cv2.imshow("Image.jpg",img)
cv2.imshow("q1.jpg",mask_res2)
cv2.imshow("q2.jpg",mask_res3)
cv2.imshow("q3.jpg",mask_res4)

print(h,w,c)
# cv2.imshow('img',img)
# cv2.imwrite("1_HSV.jpg",HSV_result)
# cv2.imwrite("3_global_result.jpg",global_result)
cv2.waitKey(0)
cv2.destroyAllWindows()  
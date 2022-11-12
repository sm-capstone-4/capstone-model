

import warnings
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from models.detector import face_detector
from models.parser import face_parser

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["SM_FRAMEWORK"] = 'tf.keras'
warnings.filterwarnings("ignore")


def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
    return im

# Test images are obtained on https://www.pexels.com/
# 이미지 로드
im = cv2.imread("images/test_min.jpg")[..., ::-1]
im = resize_image(im) # Resize image to prevent GPU OOM.
h, w, _ = im.shape
# plt.imshow(im)

# 학습 모델 로드
fd = face_detector.FaceAlignmentDetector(
    lmd_weights_path="./models/detector/FAN/2DFAN-4_keras.h5"# 2DFAN-4_keras.h5, 2DFAN-1_keras.h5
)

# 얼굴 인식
bboxes = fd.detect_face(im, with_landmarks=False)

assert len(bboxes) > 0, "No face detected."

# Display detected face
x0, y0, x1, y1, score = bboxes[0] # show the first detected face
x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])

# plt.imshow(im[x0:x1, y0:y1, :])

# face parsing 로드
prs = face_parser.FaceParser()

# 이미지 face parsing
out = prs.parse_face(im)

# plt.imshow(out[0])
# Show parsing result with annotations

# from utils.visualize import show_parsing_with_annos
# show_parsing_with_annos(out[0])

skin =  np.array(out[0])
skin_index = np.where(skin == 1)
im[skin_index] = [0,0,0]
plt.imshow(im)
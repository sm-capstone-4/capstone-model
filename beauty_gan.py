import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import cv2
import face_parsing


def beauty():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')


    def align_faces(img):
        dets = detector(img, 1)

        objs = dlib.full_object_detections()

        for detection in dets:
            s = sp(img, detection)
            objs.append(s)

        faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)

        return faces


    # test
    test_img = dlib.load_rgb_image('skin_test/12.jpg')

    test_faces = align_faces(test_img)

    fig, axes = plt.subplots(1, len(test_faces) + 1, figsize=(20, 16))
    axes[0].imshow(test_img)

    for i, face in enumerate(test_faces):
        axes[i + 1].imshow(face)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('models/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models'))
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name('X:0') # source
    Y = graph.get_tensor_by_name('Y:0') # reference
    Xs = graph.get_tensor_by_name('generator/xs:0') # output

    def preprocess(img):
        return img.astype(np.float32) / 127.5 - 1.

    def postprocess(img):
        return ((img + 1.) * 127.5).astype(np.uint8)

    img1 = dlib.load_rgb_image('skin_test/12.jpg')
    img1_faces = align_faces(img1)

    img2 = dlib.load_rgb_image('aaaaaaaaaaaaaaaaaa.jpg')
    img2_faces = align_faces(img2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    axes[0].imshow(img1_faces[0])
    axes[1].imshow(img2_faces[0])

    src_img = img1_faces[0]
    ref_img = img2_faces[0]

    X_img = preprocess(src_img)
    X_img = np.expand_dims(X_img, axis=0)

    Y_img = preprocess(ref_img)
    Y_img = np.expand_dims(Y_img, axis=0)

    output = sess.run(Xs, feed_dict={
        X: X_img,
        Y: Y_img
    })
    output_img = postprocess(output[0])

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].set_title('Source')
    axes[0].imshow(src_img)
    axes[1].set_title('Reference')
    axes[1].imshow(ref_img)
    axes[2].set_title('Result')
    axes[2].imshow(output_img)
    im = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("img/result.jpg", im)
#!/usr/bin/python
from __future__ import division
import dlib
import cv2
import numpy as np
from stuff.helper import FPS2


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('results/out.avi', fourcc, 20.0, (1280, 720))

# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture('data/short.mp4')

predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Visualize Text on OpenCV Image
def vis_text(image, string, pos):
    cv2.putText(image, string, (pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

FPS_INTERVAL = 5
fps = FPS2(FPS_INTERVAL).start()
while True:
    # here some comment
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture another frame. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    if len(dets) > 0:
        for k, d in enumerate(dets):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
            cv2.rectangle(frame, (int(d.left()/ratio), int(d.top()/ratio)),(int(d.right()/ratio), int(d.bottom()/ratio)), (0, 255, 0), 1)
    vis_text(frame, "fps: {}".format(fps.fps_local()), (10, 30))
    # print('fps:', fps.fps_local())
    # cv2.flip(frame, 180)
    cv2.imshow("image", frame)
    fps.update()
    out.write(frame)
    print('{width}:{height}'.format(width=camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    height=camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        fps.stop()
        camera.release()
        out.release()
        cv2.destroyAllWindows()
        break

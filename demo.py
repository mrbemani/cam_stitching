# -*- encoding: utf-8 -*-


import sys
import os
import time
import numpy as np
import cv2

print ("isUsingOpenCL: ", cv2.ocl.useOpenCL())
cv2.ocl.setUseOpenCL(True)
print ("OpenCL Device: ", cv2.ocl.Device_getDefault().name())

SRC_INVALID = 0
SRC_VIDEO = 1
SRC_IMAGE = 2

WINNAME = "Demo Window"
DO_STITCHING = True
SOURCE_TYPE = SRC_IMAGE
SRC_STEREO = 1

src_video_inputs = [1, 2]

src_image_list = ["images/1.jpg", "images/2.jpg", "images/3.jpg",
                  "images/4.jpg", "images/5.jpg", "images/6.jpg",
                  "images/7.jpg", "images/8.jpg", "images/9.jpg",
                  "images/10.jpg"]

use_compose = False

src_images = []
if SOURCE_TYPE == SRC_IMAGE:
    src_images = [cv2.resize(cv2.UMat(im), (int((im.shape[1]/im.shape[0])*480), 480)) for im in [cv2.imread(x) for x in src_image_list[:2]]]


cv2.namedWindow(WINNAME)
cv2.moveWindow(WINNAME, 0, 0)
#cv2.resizeWindow(WINNAME, (1280, 480))


caps = []

if SOURCE_TYPE == SRC_VIDEO:
    for c in src_video_inputs:
        print ("init video_src: [", c, "] ... ", end='', flush=True)
        caps.append(cv2.VideoCapture(c))
        print ("done!", flush=True)
        time.sleep(1)

stitcher = cv2.Stitcher_create()

im = None
k = -1
while k != ord('q'):
    frames = []
    stitching_failed = False
    h, w = 480, 640
    #print ("loading frame....")
    if SOURCE_TYPE == SRC_VIDEO:
        for cap in caps:
            ret, frame = cap.read()
            frm = None
            if ret:
                frm = frame
                h, w = frm.shape[:2]
                if h != 480:
                    frm = cv2.resize(frame, (int((w/h)*480), 480))
            else:
                frm = np.zeros((h, w * SRC_STEREO, 3), dtype=np.uint8)
            if SOURCE_TYPE == SRC_VIDEO:
                frames.append(frm)
            else:
                frames = np.split(frm, SRC_STEREO, axis=1)
    elif SOURCE_TYPE == SRC_IMAGE:
        if SRC_STEREO > 1:
            frames = np.split(src_images[0], SRC_STEREO, axis=1)
        else:
            frames = src_images
    if DO_STITCHING:
        if use_compose:
            status, stitched_image = stitcher.composePanorama(frames)
        else:
            status, stitched_image = stitcher.stitch(frames)
        if status == 0:
            im = stitched_image
            stitching_failed = False
        else:
            stitching_failed = True
    if not DO_STITCHING:# or stitching_failed:
        im = np.concatenate(frames, axis=1)
    if im is not None:
        cv2.imshow(WINNAME, im)
        print ("+", end="", flush=True)
        k = cv2.waitKey(1)
    else:
        print ("-", end="", flush=True)
        k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite("pano.jpg", im)
    if im is not None and k == ord('c'):
        if not stitching_failed:
            use_compose = True
for cap in caps:
    cap.release()
cv2.destroyAllWindows()


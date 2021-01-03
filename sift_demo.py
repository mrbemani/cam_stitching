# -*- encoding: utf-8 -*-


import sys
import os

os.environ['OPENCV_OPENCL_DEVICE'] = 'NVIDIA:dGPU:GeForce GTX 1050'

import time
from typing import List, Tuple, Dict
import numpy as np
import scipy.linalg
import cv2

print ("isUsingOpenCL: ", cv2.ocl.useOpenCL())
cv2.ocl.setUseOpenCL(True)
print ("OpenCL Device: ", cv2.ocl.Device_getDefault().name())




SCREEN_W, SCREEN_H = 1024, 640

#cv2.

SRC_INVALID = 0
SRC_VIDEO = 1
SRC_IMAGE = 2

WINNAME = "Demo Window"
DO_STITCHING = True
SOURCE_TYPE = SRC_IMAGE
SRC_STEREO = 1
MIN_MATCH_COUNT = 50
IMAGE_H = 480

src_video_inputs = [2, 1]
src_image_list = ["images/{}.jpg".format(i) for i in range(1, 11)]

base_view_idx = 5

src_images = []
if SOURCE_TYPE == SRC_IMAGE:
    src_images = [cv2.resize(cv2.UMat(im), (int((im.shape[1]/im.shape[0])*IMAGE_H), IMAGE_H)) for im in [cv2.imread(x) for x in src_image_list]]

cv2.namedWindow(WINNAME)
cv2.moveWindow(WINNAME, 0, 0)
cv2.resizeWindow(WINNAME, (SCREEN_W, SCREEN_H))


caps = []

if SOURCE_TYPE == SRC_VIDEO:
    for c in src_video_inputs:
        print ("init video_src: [", c, "] ... ", end='', flush=True)
        caps.append(cv2.VideoCapture(c))
        print ("done!", flush=True)
        time.sleep(1)


def stitch_images(images: List):
    sift = cv2.SIFT_create()
    kps = []
    descs = []
    ims = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    for im in ims:
        kp, desc = sift.detectAndCompute(im, None)
        kps.append(kp)
        descs.append(desc)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 80)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    match_list = []
    for _idx in range(len(descs)):
        for _jdx in range(len(descs)):
            if _jdx > _idx:
                matches = flann.knnMatch(descs[_idx], descs[_jdx], k = 2)
                good_matches = []
                for m in matches:
                    if m[0].distance < 0.7*m[1].distance:
                        good_matches.append(m[0])
                m = [_idx, _jdx, good_matches, kps[_idx], kps[_jdx]]
                match_list.append(m)
    top_match_list = sorted(match_list, key=lambda x: len(x[2]), reverse=True)
    for _m_idx in range(len(top_match_list)):
        _idx, _jdx, _good, kp1, kp2 = top_match_list[_m_idx]
        if len(_good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in _good ]).reshape(-1, 1, 2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in _good ]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            top_match_list[_m_idx].append(M)
            top_match_list[_m_idx].append(matchesMask)
    return top_match_list

def computeMToIdx(idx, matchList: List):
    M_Paths = []
    added_matches = []
    while len(added_matches) < len(matchList):
        _a_m_start_len = len(added_matches)
        for _i in range(len(matchList)):
            if _i in added_matches:
                continue
            m = matchList[_i]
            if m[0] == idx:
                added_matches.append(_i)
                M_Paths.append([m[1], m[0]])
            elif m[1] == idx:
                added_matches.append(_i)
                M_Paths.append([m[0], m[1]])
        if len(added_matches) > _a_m_start_len: continue
        M_Paths.sort(key=lambda k: len(k))
        for _i in range(len(matchList)):
            if _i in added_matches:
                continue
            m = matchList[_i]
            if idx not in m[:2]:
                last_idx_s = [x[0] for x in M_Paths]
                z0 = -1
                z1 = -1
                try:
                    z0 = last_idx_s.index(m[0])
                except:
                    z0 = -1
                try:
                    z1 = last_idx_s.index(m[1])
                except:
                    z1 = -1
                if z0 != -1 and z1 == -1:
                    added_matches.append(_i)
                    M_Paths.append([m[1], *M_Paths[z0]])
                elif z1 != -1 and z0 == -1:
                    added_matches.append(_i)
                    M_Paths.append([m[0], *M_Paths[z1]])
            M_Paths.sort(key=lambda k: len(k))
        _a_m_end_len = len(added_matches)
        addedCount = _a_m_end_len - _a_m_start_len
        if addedCount < 1: break
    M_Matrices = dict()
    H_s = dict()
    for m in matchList:
        if len(m) > 5:
            H_s["{}_{}".format(m[0], m[1])] = m[5]
            H_s["{}_{}".format(m[1], m[0])] = np.linalg.inv(m[5])
    for p in M_Paths:
        k = p[0]
        M_s = []
        for i in range(len(p)-1):
            H_s_key = "{}_{}".format(p[i], p[i+1])
            if H_s_key in H_s:
                H = H_s[H_s_key]
                M_s.append(H)
        M = np.identity(3)
        for h in M_s:
            M = h.dot(M)
        M_Matrices[k] = M
    return M_Matrices

match_params = []    
M_trans = [] 
imasks = dict()  

im = None
k = -1
m_i = 0
while k != ord('q'):
    frames = []
    stitching_failed = False
    h, w = IMAGE_H, int(IMAGE_H*(4./3.))
    #print ("loading frame....")
    if SOURCE_TYPE == SRC_VIDEO:
        for cap in caps:
            ret, frame = cap.read()
            frm = None
            if ret:
                frm = frame
                h, w = frm.shape[:2]
                if h != IMAGE_H:
                    frm = cv2.resize(frame, (int((w/h)*IMAGE_H), IMAGE_H))
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
        if len(match_params) < 1:
            print ("calculating views...", flush=True)
            match_params = stitch_images(frames)
            M_trans = computeMToIdx(base_view_idx, match_params)
        im = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        im_mask = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)
        for st_im_idx in range(len(frames)):
            im_obj = []
            frm = frames[st_im_idx]
            result = None
            M = np.identity(3)
            if st_im_idx != base_view_idx: 
                M = M_trans[st_im_idx]
                if M.all() == np.identity(3).all():
                    continue
            #else:
            #    continue
            if np.linalg.det(M) <= 0.05:
                continue
            #K = np.array([[1.3, 0, SCREEN_W/2, 0], [0, 1.3, SCREEN_H/2, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float)
            #K_RT = np.array([[1, 0, 0, 0], [0, np.cos(r_i), -np.sin(r_i), 0], [0, np.sin(r_i), np.cos(r_i), 0], [0, 0, 0, 1]])
            #R = np.array([[1., 0, 0], [0, np.cos(r_i), np.sin(r_i)], [0, -np.sin(r_i), np.cos(r_i)]], dtype=np.float)
            T = np.array([[1, 0, (SCREEN_W-w)/2], [0, 1, (SCREEN_H-h)/2], [0, 0, 1]], dtype=np.float)
            T_inv = np.linalg.inv(T)
            #M = T.dot(M)
            #M = np.pad(M, (0,1))
            #M[3, 3] = 1
            #T_R_M = K.dot(K_RT).dot(M)#dot(T.dot(R).dot(T_inv).dot())
            #print (M)
            #print (T_R_M)
            TRM = T.dot(M)
            if np.linalg.det(TRM) <= 0.05: 
                print("-[{}]-".format(st_im_idx), end="", flush=True)
                continue
            result = cv2.warpPerspective(frm, TRM, (SCREEN_W, SCREEN_H))
            #if True:
            if st_im_idx not in imasks:
                mask = result.get().copy()
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask[mask!=0] = 255
                kernel = np.ones((3, 3),np.uint8)
                mask = cv2.erode(cv2.UMat(mask), kernel, 1)
                imasks[st_im_idx] = cv2.bitwise_not(im_mask)
                im_mask = cv2.bitwise_or(im_mask, mask)
            cropped_result = cv2.bitwise_and(result, (255, 255, 255), mask=imasks[st_im_idx])
            #if st_im_idx == base_view_idx:
            im = cv2.add(im, cropped_result)

        print ("+", end="", flush=True)
        m_i += 1
    if not DO_STITCHING:# or stitching_failed:
        im = np.concatenate(frames, axis=1)
    if im is not None:
        cv2.imshow(WINNAME, im)
        k = cv2.waitKey(1)
    else:
        k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite("saved2.jpg", im)
for cap in caps:
    cap.release()
cv2.destroyAllWindows()


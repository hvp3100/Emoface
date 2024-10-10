import numpy as np
# import torch
import pickle as pkl
from gdl.utils.FaceDetector import FaceDetector
import os, sys
# from gdl.utils.other import get_path_to_externals 
from pathlib import Path

import mediapipe as mp
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark


def mediapipe2np(landmarks): 
    # d = protobuf_to_dict(landmarks)
    array = np.zeros(shape=(len(landmarks), 3))
    for i in range(len(landmarks)):
        array[i, 0] = landmarks[i].x
        array[i, 1] = landmarks[i].y
        array[i, 2] = landmarks[i].z
    return array


def np2mediapipe(array): 

    landmarks = NormalizedLandmarkList()
    for i in range(len(array)):
        if array.shape[1] == 3:
            lmk = NormalizedLandmark(x=array[i, 0], y=array[i, 1], z=array[i, 2])
        else: 
            lmk = NormalizedLandmark(x=array[i, 0], y=array[i, 1], z=0.)
        landmarks.landmark.append(lmk)
    return landmarks


class MediaPipeLandmarkDetector(FaceDetector):

    def __init__(self, threshold=0.1, max_faces=1, video_based=False):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        # self.mp_face_mesh_options = mp.FaceMeshCalculatorOptions()

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=not video_based,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=threshold)
    
    def run(self, image, with_landmarks=False, detected_faces=None):

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            results = self.face_mesh.process(image) 


        if not results.multi_face_landmarks:
            print("no face detected by mediapipe")
            if with_landmarks:
                return [],  'mediapipe', [] 
            else:
                return [],  'mediapipe'


        all_landmarks = []
        all_boxes = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = mediapipe2np(face_landmarks.landmark)

            landmarks = landmarks * np.array([image.shape[1], image.shape[0], 1])
            
            all_landmarks += [landmarks]

            left = np.min(landmarks[:, 0])
            right = np.max(landmarks[:, 0])
            top = np.min(landmarks[:, 1])
            bottom = np.max(landmarks[:, 1])

            bbox = [left, top, right, bottom]
            all_boxes += [bbox]

        if with_landmarks:
            return all_boxes, 'mediapipe', all_landmarks
        else:
            return all_boxes, 'mediapipe'



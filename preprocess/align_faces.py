# import the necessary packages
from face_alignment import FaceAligner
import dlib
import cv2
from glob import glob
import subprocess
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor)

dataDir = "../../GRID/data"
outDir = "../mouth-data"

for i in range(1, 35):
    if i == 21:
        continue
    for filename in glob(f"{dataDir}/s{i}/*.mpg"):

        cap = cv2.VideoCapture(filename)
        fcc = cv2.VideoWriter_fourcc(*'X264')

        stem = filename[:-4]
        out = cv2.VideoWriter(f"{outDir}/s{i}/{stem}_lips.avi", fcc, 25.0, (100, 50))

        ret, frame = cap.read()
        rects = detector(frame, 0)

        # If there's no face, this video is corrupted
        if len(rects) == 0:
            cap.release()
            out.release()
            subprocess.run(["rm", filename])
            continue

        frames = []
        while ret:
            frames.append(frame)
            ret, frame = cap.read()

        # Get the alignment for the middle frame
        middleFrame = frames[len(frames) // 2]
        rects = detector(middleFrame, 0)
        if len(rects) == 0:
            continue
        M = fa.getAlignment(middleFrame, rects[0])

        # Write aligned frames to file
        for frame in frames:
            out.write(fa.transform(frame, M))

        cap.release()
        out.release()


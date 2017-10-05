# import the necessary packages
from face_alignment import FaceAligner
import dlib
import cv2
from glob import glob
import subprocess
from scipy.interpolate import interp1d
from shutil import copyfile

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor)

dataDir = "../../GRID/data"
labelDir = "../../GRID/trans"
outDir = "../mouth-data"
outlabel = "../mouth-label"

for i in range(1, 35):
    if i == 21:
        continue
    for filename in glob(f"{dataDir}/s{i}/*.mpg"):

        cap = cv2.VideoCapture(filename)
        fcc = cv2.VideoWriter_fourcc(*'X264')

        stem = filename[-10:-4]

        frames = []
        ret, frame = cap.read()
        while ret:
            frames.append(frame)
            ret, frame = cap.read()

        with open(f"{labelDir}/s{i}/{stem}.align") as f:
            start, stop, word = f.readline().split()
            while word in ["sil", "sp"]:
                start, stop, word = f.readline().split()
            begin = int(start) // 1000
            while word != "sil":
                start, stop, word = f.readline().split()
            end = int(start) // 1000

        frameList = [0, begin, (begin + end) // 2, end, 74]

        Ms = []
        for index in frameList:
            if index >= len(frames):
                break
            rects = detector(frames[index], 0)
            if len(rects) == 0:
                break
            M = fa.getAlignment(frames[index], rects[0])

            Ms.append(M)

        if len(Ms) != 5:
            continue

        copyfile(f"{labelDir}/s{i}/{stem}.align", f"{outlabel}/s{i}_{stem}.align")

        f = interp1d(frameList, Ms, axis=0)

        out = cv2.VideoWriter(f"{outDir}/s{i}_{stem}.avi", fcc, 25.0, (100, 50))

        # Write aligned frames to file
        for index, frame in enumerate(frames):
            out.write(fa.transform(frame, f(index)))

        cap.release()
        out.release()


import pandas as pd
import cv2
import numpy as np
import csv
import dlib


def get_directory(label, file):
    switcher = {
        0: "./dataset/cropped_eyes_emotions/angry/",
        1: "./dataset/cropped_eyes_emotions/disgust/",
        2: "./dataset/cropped_eyes_emotions/fear/",
        3: "./dataset/cropped_eyes_emotions/happy/",
        4: "./dataset/cropped_eyes_emotions/sad/",
        5: "./dataset/cropped_eyes_emotions/surprise/",
        6: "./dataset/cropped_eyes_emotions/neutral/"
    }
    out = switcher.get(label) + file
    return out


def convert_data(pixels):
    pixels = [int(pixel) for pixel in pixels.split(' ')]
    pixels = np.array(pixels, dtype=np.uint8).reshape(48, 48)
    return pixels


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

outFile = open("./cropped_eyes_dataset.csv", "w")
writer = csv.writer(outFile)
writer.writerow(["emotion", "pixels"])

data = pd.read_csv("./fer2013.csv")
data = data.drop('Usage', 1)
values = np.array(data['pixels'])
labels = np.array(data['emotion'])

csvOut = []
c = 0
for i in range(0, len(values)):
    face = convert_data(values[i])
    rects = detector(face, 1)
    for (a, rect) in enumerate(rects):
        print(rect)
        shape = predictor(face, rect)

        x = shape.part(0).x
        w = shape.part(16).x
        y = max(shape.part(19).y, shape.part(24).y)
        h = max(shape.part(1).y, shape.part(15).y)

        y = max(0, (y - (h - y)))
        x = max(0, x)
        print(x, y, w, h)
        crop = face[y:h, x:w]
        cropString = ""
        for k in range(0, len(crop)):
            for j in range(0, len(crop[0])):
                cropString = cropString + " " + str(crop[k][j])
        fileName = str(i) + "_" + str(a) + ".jpg"
        fileName = get_directory(labels[i], fileName)
        if len(crop) != 0:
            cv2.imwrite(fileName, crop)
            csvOut.append([labels[i], cropString])
            c = c+1
print(c)
writer.writerows(csvOut)

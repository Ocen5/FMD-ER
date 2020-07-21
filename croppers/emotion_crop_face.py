import pandas as pd
import cv2
import dlib
import numpy as np
import csv


def get_directory(label, file):
    switcher = {
        0: "./dataset/cropped_face_emotions/angry/",
        1: "./dataset/cropped_face_emotions/disgust/",
        2: "./dataset/cropped_face_emotions/fear/",
        3: "./dataset/cropped_face_emotions/happy/",
        4: "./dataset/cropped_face_emotions/sad/",
        5: "./dataset/cropped_face_emotions/surprise/",
        6: "./dataset/cropped_face_emotions/neutral/"
    }
    out = switcher.get(label) + file
    return out


def convert_data(pixels):
    pixels = [int(pixel) for pixel in pixels.split(' ')]
    pixels = np.array(pixels, dtype=np.uint8).reshape(48, 48)
    return pixels


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

outFile = open("./cropped_face_dataset.csv", "w")
writer = csv.writer(outFile)
writer.writerow(["emotion", "pixels"])

data = pd.read_csv("./fer2013.csv")
data = data.drop('Usage', 1)
values = np.array(data['pixels'])
labels = np.array(data['emotion'])

csvOut = []
for i in range(0, len(values)):
    face = convert_data(values[i])
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # image = Image.fromarray(image)
    rects = detector(face, 1)
    for (a, rect) in enumerate(rects):
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()
        if x > 0 and y > 0:
            print(x, y, w, h)
            crop = face[y:h, x:w]
            cropString = ""
            for k in range(0, len(crop)):
                for j in range(0, len(crop[0])):
                    cropString = cropString + " " + str(crop[k][j])
            fileName = str(i) + "_" + str(a) + ".jpg"
            fileName = get_directory(labels[i], fileName)
            cv2.imwrite(fileName, crop)
            csvOut.append([labels[i], cropString])
writer.writerows(csvOut)

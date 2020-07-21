import pandas as pd
import cv2
import numpy as np
import dlib


def get_directory(label, file, position):
    switcher = {
        0: "angry/",
        1: "disgust/",
        2: "fear/",
        3: "happy/",
        4: "sad/",
        5: "surprise/",
        6: "neutral/"
    }
    out = "./dataset/cropped_" + position + "_emotion/" + switcher.get(label) + file
    return out


def convert_data(pixels):
    pixels = [int(pixel) for pixel in pixels.split(' ')]
    pixels = np.array(pixels, dtype=np.uint8).reshape(48, 48)
    return pixels


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

data = pd.read_csv("./dataset/fer2013.csv")
data = data.drop('Usage', 1)
values = np.array(data['pixels'])
labels = np.array(data['emotion'])

for i in range(0, len(values)):
    face = convert_data(values[i])
    rects = detector(face, 1)
    for (a, rect) in enumerate(rects):
        print(i+1)
        print(rect)
        shape = predictor(face, rect)

        x = max(0, shape.part(0).x)
        w = shape.part(27).x
        y = max(shape.part(17).y, shape.part(21).y)
        h = max(shape.part(1).y, shape.part(28).y)
        left_eye = face[y:h, x:w]
        print(x, w, y, h)

        x = max(0, shape.part(27).x)
        w = shape.part(16).x
        y = max(shape.part(22).y, shape.part(26).y)
        h = max(shape.part(15).y, shape.part(28).y)
        right_eye = face[y:h, x:w]
        print(x, w, y, h)

        x = max(0, shape.part(40).x)
        w = shape.part(47).x
        y = max(shape.part(38).y, shape.part(43).y)
        h = shape.part(33).y
        nose = face[y:h, x:w]
        print(x, w, y, h)

        x = max(0, shape.part(3).x)
        w = shape.part(13).x
        y = shape.part(33).y
        h = shape.part(8).y
        mouth = face[y:h, x:w]
        print(x, w, y, h)

        x = max(0, shape.part(0).x)
        w = shape.part(16).x
        y = max(shape.part(19).y, shape.part(24).y)
        h = max(shape.part(1).y, shape.part(15).y)
        h, y = y, max(0, (y - (h - y)))
        front = face[y:h, x:w]
        print(x, w, y, h)

        '''
        x = max(0, rect.left())
        y = rect.top()
        w = rect.right()
        h = rect.bottom()
        full_face = image[y:h, x:w]
        print(x,w,y,h)
        '''

        fileName = str(i) + "_" + str(a) + "left_eye.jpg"
        fileName = get_directory(labels[i], fileName, "left_eye")
        if left_eye.size:
            cv2.imwrite(fileName, left_eye)

        fileName = str(i) + "_" + str(a) + "right_eye.jpg"
        fileName = get_directory(labels[i], fileName, "right_eye")
        if right_eye.size:
            cv2.imwrite(fileName, right_eye)

        fileName = str(i) + "_" + str(a) + "mouth.jpg"
        fileName = get_directory(labels[i], fileName, "mouth")
        if mouth.size:
            cv2.imwrite(fileName, mouth)

        fileName = str(i) + "_" + str(a) + "nose.jpg"
        fileName = get_directory(labels[i], fileName, "nose")
        if nose.size:
            cv2.imwrite(fileName, nose)

        fileName = str(i) + "_" + str(a) + "front.jpg"
        fileName = get_directory(labels[i], fileName, "front")
        if front.size:
            cv2.imwrite(fileName, front)

        '''
        fileName = str(i) + "_" + str(a) + "full.jpg"
        fileName = get_directory(labels[i], fileName)
        if full_face.size:
            cv2.imwrite(fileName, full_face)
        '''

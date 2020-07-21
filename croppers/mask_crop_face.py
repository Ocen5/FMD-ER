import cv2
import dlib
import numpy as np
from imutils import paths

maskImages = paths.list_images("./dataset/with_mask/")
noMaskImages = paths.list_images("./dataset/without_mask/")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

i = 0
for face in noMaskImages:
    print(face)
    find = False
    img = cv2.imread(face, cv2.IMREAD_COLOR)
    rects = detector(img, 1)
    print(rects)
    for (a, rect) in enumerate(rects):
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()

        if x > 0 and y > 0:
            crop = img[y:h, x:w]
            fileName = "./dataset/cropped_without_mask/" + str(i) + "_" + str(a) + ".jpg"
            cv2.imwrite(fileName, crop)

        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        shape = predictor(img, rect)                            # il metodo utilizzato in questo modulo è analogo a quello usato in
        coords = np.zeros((68, 2), dtype="int")                 # mask_crop_eyes, ma qui, oltre che alle immagini ritagliate sulla zona
        for j in range(0, 68):                                  # del volto, vengono salvate anche le intere immagini con l'aggiunta del
            coords[j] = (shape.part(j).x, shape.part(j).y)      # riquadro che specifica il volto e i landmarks facciali. Queste ultime
        shape = coords                                          # immagini sono servite come feedback ai docenti
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        find = True
    if find:
        fileName = "./dataset/pointed_images/noMask/00" + str(i) + "_" + str(a) + ".jpg"
        cv2.imwrite(fileName, img)
    i = i + 1

i = 0
for face in maskImages:
    print(face)
    find = False
    img = cv2.imread(face, cv2.IMREAD_COLOR)
    rects = detector(img, 1)
    print(rects)
    for (a, rect) in enumerate(rects):
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()

        if x > 0 and y > 0:
            crop = img[y:h, x:w]
            fileName = "./dataset/cropped_with_mask/" + str(i) + "_" + str(a) + ".jpg"
            cv2.imwrite(fileName, crop)

        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        shape = predictor(img, rect)
        coords = np.zeros((68, 2), dtype="int")
        for j in range(0, 68):
            coords[j] = (shape.part(j).x, shape.part(j).y)
        shape = coords
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        find = True
    if find:
        fileName = "./dataset/pointed_images/mask/00" + str(i) + ".jpg"
        cv2.imwrite(fileName, img)
    i = i + 1

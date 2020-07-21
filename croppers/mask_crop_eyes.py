import cv2
import dlib
from imutils import paths

'''
QUESTO MODULO VIENE ESEGUITO SUL DATASET DELLA MASCHERINA COME TEST
VERRA' UTILIZZATO PER RITAGLIARE LA PORZIONE DEGLI OCCHI SUL DATASET FER2013
'''
# inizializzazione del face detector e scelta delle cartelle del dataset
maskImages = paths.list_images("./dataset/not_cropped/with_mask/")
noMaskImages = paths.list_images("./dataset/not_cropped/without_mask/")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

i = 0
for face in noMaskImages:
    print(face)
    img = cv2.imread(face, cv2.IMREAD_COLOR)   #acquisizione immagine
    rects = detector(img, 1)                    #rilevamento volti nell'immagine
    print(rects)
    for (a, rect) in enumerate(rects):
        shape = predictor(img, rect)            #predizione dei landmarks per ogni volto nell'immagine

        x = max(0, shape.part(0).x)                     #acquisizione dei landmarks che corrispondono alla porzione degli occhi
        w = shape.part(16).x
        y = max(shape.part(19).y, shape.part(24).y)     #acquisizione della massima cordinata Y tra i landmarks 19-24 e 1-15
        h = max(shape.part(1).y, shape.part(15).y)      #in modo da evitare che il ritaglio di volti inclinati possa eliminare dati necessari

        y = max(0, (y - (h - y)))               # diminuisce la dimensione di y per cercare di prendere anche la zona della fronte.
                                                # viene usato il max tra 0 e y per assicurarsi che y rimanga nell'immagine

        print(x, w, y, h)
        crop = img[y:h, x:w]                    # ritaglio della porzione di immagine selezionanta
        fileName = "./dataset/cropped_eyes/" + str(i) + "_" + str(a) + ".jpg"
        cv2.imwrite(fileName, crop)
    i = i + 1

for face in maskImages:                        # ragionamento analogo al precedente su directory diversa
    print(face)
    img = cv2.imread(face, cv2.IMREAD_COLOR)
    rects = detector(img, 1)
    print(rects)
    for (a, rect) in enumerate(rects):
        shape = predictor(img, rect)

        x = max(0, shape.part(0).x)
        w = shape.part(16).x
        y = max(shape.part(19).y, shape.part(24).y)
        h = max(shape.part(1).y, shape.part(15).y)

        y = max(0, (y - (h - y)))  # diminuisce la dimensione di y per cercare di prendere anche la zona della fronte.
                                   # viene usato il max tra 0 e y per assicurarsi che y rimanga nell'immagine

        print(x, w, y, h)
        crop = img[y:h, x:w]
        fileName = "./dataset/cropped_eyes/" + str(i) + "_" + str(a) + ".jpg"
        cv2.imwrite(fileName, crop)
    i = i + 1


import numpy as np
from imutils import paths
from keras.applications.vgg16 import preprocess_input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.engine import  Model
from keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace

num_features = 64

batch_size = 64
epochs = 50
width, height = 48, 48
nb_class = 7
hidden_dim = 512

def getLabel(path):
    if "angry" in path:
        return 0
    if "disgust" in path:
        return 1
    if "fear" in path:
        return 2
    if "happy" in path:
        return 3
    if "neutral" in path:
        return 4
    if "sad" in path:
        return 5
    return 6


imagePaths = list(paths.list_images("D:/Users/Marco/Desktop/cropped_eyes_emotions/cropped_face_emotions/"))
data = []
labels = []

for imagePath in imagePaths:
    print(imagePath)
    label = getLabel(imagePath)

    image = load_img(imagePath, target_size=(48, 48))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.1, stratify=labels)
trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.1, random_state=41)

trainX = np.asarray(trainX, dtype=np.uint8)
testX = np.asarray(testX, dtype=np.uint8)

trainY = np.asarray(trainY, dtype=np.uint8)
testY = np.asarray(testY, dtype=np.uint8)

validX = np.asarray(validX, dtype=np.uint8)
validY = np.asarray(validY, dtype=np.uint8)

print(trainX)
print(trainY)

trainY = to_categorical(trainY)
testY = to_categorical(testY)
validY = to_categorical(validY)

vgg_model = VGGFace(include_top=False, input_shape=(48, 48, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)

for layer in custom_vgg_model.layers[:-4]:
  layer.trainable = False

custom_vgg_model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

history = custom_vgg_model.fit(trainX, trainY,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(validX, validY),
                    shuffle=True)

print("Saved model to disk")
custom_vgg_model.save("prima_rete.h5")

print("[INFO] evaluating network...")
predIdxs = custom_vgg_model.predict(testX, batch_size=batch_size)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs))

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["accuracy"],'b', label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"],'r',label="valid_acc")
plt.title("Training and Validation accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")

# plot training loss
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"],'b', label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"],'r', label="valid_loss")
plt.title("Training and Validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot1")



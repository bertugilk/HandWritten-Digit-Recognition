import numpy as np
import cv2
from keras.models import load_model

model=load_model("Model/model.h5")
image=cv2.imread('Test Images/6.png')

def preProcessing(img):
    newImage = np.array(img)
    newImage = cv2.resize(newImage, (28, 28))
    newImage = newImage.flatten()
    newImage = newImage.reshape(1, 28, 28, 1)
    return newImage

def thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh1

image=thresh(image)
img=preProcessing(image)

classIndex = int(model.predict_classes(img))
predictions = model.predict(img)
probVal = np.amax(predictions)
print(classIndex, probVal)
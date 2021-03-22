import numpy as np
import cv2
from keras.models import load_model

model=load_model("Model/model.h5")
threshold=0.65
camera=cv2.VideoCapture(0)

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

def main():
    while True:
        ret,frame=camera.read()

        x, y, w, h = (0, 0, 280, 300)
        cut = frame[y:y + h, x:x + w]

        image=thresh(cut)
        cv2.imshow("image",image)
        img = preProcessing(image)

        classIndex = int(model.predict_classes(img))
        predictions = model.predict(img)
        probVal = np.amax(predictions)
        print(classIndex, probVal)

        if probVal > threshold:
            cv2.putText(frame, "Digit: " + str(classIndex), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Accuracy: " + str(probVal), (5, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Frame",cut)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('model.h5')
categories = ['happiness', 'neutral', 'sadness']


def detect_faces(image):
    """
    This function detects faces using a cascade classifier
    """
    image_copy = image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    return faces


def contour_face(image, x, y, w, h):
    """
    This function draws a rectangle around the face
    """
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


def crop_face(image, x, y, w, h):
    """
    This function crops the face from the image
    """
    return image[y:y+h, x:x+w]


def predict_emotion(face):
    """
    This function transforms the image and uses the model to predict the emotions
    """
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray, (48, 48))
    resized_face = resized_face.reshape(-1, 48, 48, 1)
    prediction = model.predict(resized_face)
    pred_class = np.argmax(prediction, axis=1)
    emotion = categories[pred_class[0]]

    return emotion


def main():
    """
    This is a main function that displays the image and triggers other functions
    """
    cam_test = cv2.VideoCapture(0)
    while True:
        _, img = cam_test.read()
        faces = detect_faces(img)
        for (x, y, w, h) in faces:
            contour_face(img, x, y, w, h)
            face = crop_face(img, x, y, w, h)
            emotion = predict_emotion(face)
            cv2.putText(img, emotion, (int((x+h)/2), y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:  # ESC
            break
    cv2.destroyAllWindows()


def predict_samples():
    for i in range(3):
        img = cv2.imread(f'sample_pictures/0{i+1}.png')
        faces = detect_faces(img)
        for (x, y, w, h) in faces:
            contour_face(img, x, y, w, h)
            face = crop_face(img, x, y, w, h)
            emotion = predict_emotion(face)
            cv2.putText(img, emotion, (int((x+h)/2), y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow(f'img{i+1}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # predict_samples()
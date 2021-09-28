import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def display_camera():
    """
    This function captures the image from the camera and displays it. ESC button stops the loop
    """
    cam = cv2.VideoCapture(0)

    while True:
        _, image = cam.read()
        faces = detect_faces(image)
        image_cont = contour_faces(image, faces)
        cv2.imshow('my cam', image_cont)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


def get_one_frame():
    """
    This function captures one frame
    """
    cam = cv2.VideoCapture(0)
    _, image = cam.read()
    faces = detect_faces(image)
    image_cont = contour_faces(image, faces)
    plt.imshow(image_cont)
    plt.show()

    cv2.destroyAllWindows()


def detect_faces(image):
    """
    This function detects faces using a cascade classifier
    """
    image_copy = image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    return faces


def contour_faces(image, faces):
    """
    This function draws rectangles around the faces
    """
    image_copy = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image_copy


def test_cropping():
    """
    This is a test function to crop the face. It'll be removed later
    """
    cam_test = cv2.VideoCapture(0)
    while True:
        _, img = cam_test.read()
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        faces = detect_faces(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            cv2.imshow('face', face)
        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:  # ESC
            break

if __name__ == '__main__':
    # display_camera()
    # get_one_frame()
    test_cropping()
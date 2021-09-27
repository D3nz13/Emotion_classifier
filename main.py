import cv2
import matplotlib.pyplot as plt

def display_camera():
    """
    This function captures the image from the camera and displays it. ESC button stops the loop
    """
    cam = cv2.VideoCapture(0)

    while True:
        _, image = cam.read()
        cv2.imshow('my cam', image)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


def get_one_frame():
    """
    This function captures one frame
    """
    cam = cv2.VideoCapture(0)
    _, image = cam.read()
    plt.imshow(image)
    plt.show()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    display_camera()
    # get_one_frame()
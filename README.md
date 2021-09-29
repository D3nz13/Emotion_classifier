# Emotion Classifier
## Table of conents
* [Description](#description)
* [Used libraries](#used-libraries)
* [Neural network architecture](#neural-network-architecture)
* [Graphs](#graphs-and-images)
* [Example of use](#example-of-use)
* [Possible improvements](#possible-improvements)

## Description
This project has been made to learn the basics of the OpenCV library.  
I used a dataset from kaggle ([dataset](https://www.kaggle.com/deadskull7/fer2013)) to build a neural network to recognize emotions. The dataset contained 48x48 grayscale images of faces and 7 labels. To simplify the task, I used only 3 labels - happiness, neutral, sadness. The neural network was trained and saved, the model is saved in this directory (```model.h5```)  
The program captures the video from a camera and then processes each frame.
1) face detection using face_cascade_classifier ([available here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml))  
1.1) before the classifier is applied, the image is converted to grayscale  
2) for each face the program draws contours around the face
3) for each face the program crops the image to obtain only the face
4) on every cropped image of a face, the neural network predicts the most probable emotion  
4.1) before the model gets the image, the cropped image needs to be converted to grayscale and resized to 48x48 pixels  
5) having predicted the emotion, the program puts a label representing the emotion above the face  
6) the program can be stopped by clicking the ESC button

## Used libraries
```tensorflow==2.5.0```  
```numpy==1.19.5```  
```opencv-python==4.5.3.56```

## Neural network architecture
![neural-network-architecture](/graphs/summary.png)

## Graphs and images
- Training and validation loss/accuracy  
![training](/graphs/training.png)  
- Sample predictions from the validation dataset  
![val_preds](/graphs/model_sample_predictions.png)  

## Example of use
- the sample pictures have been downloaded from ```google pictures```  
![](/graphs/sample_pic_1_pred.png)  
![](/graphs/sample_pic_2_pred.png)  
![](/graphs/sample_pic_3_pred.png)  
![](/graphs/sample_pic_4_pred.png)  

## Possible improvements
- include all 7 emotions from the dataset, not only 3  
- improve the neural network, right now it sometimes struggles to recognize sadness  
- tune the haarcascade classifier parameters, sometimes it considers something a face while it's not a face
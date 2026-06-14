#!/usr/bin/env python
# coding: utf-8

# #### Group-12 MLC Project Codes

# Importing The Required Libraries

# In[1]:


import os
import warnings
import logging
from absl import logging as absl_logging
import tensorflow as tf
import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# - Loads the MNIST handwritten digit dataset and preprocesses the images for training.
# - Builds and trains a Convolutional Neural Network (CNN) model for digit recognition.
# - Saves the trained model as Models/Digit_Model.keras for future predictions.

# In[2]:


# Must be set BEFORE tensorflow import
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

warnings.filterwarnings('ignore')

# Silence tensorflow logs
logging.disable(logging.WARNING)

# Silence absl logs
absl_logging.set_verbosity('error')

# Disable tensorflow logger
tf.get_logger().setLevel('ERROR')

# Loading already trained model
loadedModel=tf.keras.models.load_model("Model/Digit_Model.keras")
print("Model Loaded Successfully")


# Performing The OCR Scan

# In[5]:


# OCR Scanner
def performOCR(imagePath):
    image=cv2.imread(imagePath)
    if image is None:
        return None
    grayImage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurredImage=cv2.GaussianBlur(grayImage,(3,3),0)
    _,thresholdImage=cv2.threshold(blurredImage,120,255,cv2.THRESH_BINARY_INV)
    extractedText=pytesseract.image_to_string(thresholdImage,config='--psm 7')
    extractedText=extractedText.strip()
    extractedText=extractedText.replace(" ","")
    return extractedText


# Digit Segmentation (Multiple Digits)

# - Processes the uploaded image using OpenCV techniques like thresholding and contour detection.
# - Identifies and extracts multiple handwritten digits from a single image.
# - Resizes and formats each detected digit into a standard 28×28 format for prediction.

# In[6]:


# Function to auto-correct rotated images
def fixRotation(imageData):
    points=np.column_stack(np.where(imageData > 0))

    # No foreground found
    if len(points)==0:
        return imageData
    angle=cv2.minAreaRect(points)[-1]

    # Correct OpenCV angle format
    if angle < -45:
        angle=-(90 + angle)
    else:
        angle=-angle

    # IMPORTANT SAFETY CHECK
    # Ignore tiny rotations
    # Prevents distortion of digits like 1
    if abs(angle)<8:
        return imageData
    imageHeight,imageWidth=imageData.shape[:2]
    imageCenter=(imageWidth//2,imageHeight//2)
    rotationMatrix=cv2.getRotationMatrix2D(imageCenter,angle,1.0)
    rotatedImage=cv2.warpAffine(imageData,rotationMatrix,(imageWidth,imageHeight),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return rotatedImage


# Digit Preparation

# In[7]:


# PREPARE DIGIT

def prepareDigit(digitImage):

    # Finding exact digit area
    coordinates=cv2.findNonZero(digitImage)

    if coordinates is None:
        return np.zeros((28,28),dtype="float32")

    xPos,yPos,boxWidth,boxHeight=cv2.boundingRect(
        coordinates
    )

    # Cropping extra black area
    croppedDigit=digitImage[
        yPos:yPos+boxHeight,
        xPos:xPos+boxWidth
    ]

    h,w=croppedDigit.shape

    # Preserve aspect ratio
    if h>w:

        newH=20

        newW=max(
            1,
            int(w*20/h)
        )

    else:

        newW=20

        newH=max(
            1,
            int(h*20/w)
        )

    # Resize properly
    resizedDigit=cv2.resize(
        croppedDigit,
        (newW,newH)
    )

    # Creating black canvas
    canvas=np.zeros(
        (28,28),
        dtype=np.uint8
    )

    # Centering digit
    startX=(28-newW)//2
    startY=(28-newH)//2

    canvas[
        startY:startY+newH,
        startX:startX+newW
    ]=resizedDigit

    # Smooth edges slightly
    canvas=cv2.GaussianBlur(
        canvas,
        (3,3),
        0
    )

    # Normalize
    canvas=canvas.astype(
        "float32"
    )/255.0

    return canvas


# Function For Extracting The Digits

# In[8]:


# EXTRACT DIGITS
def extractDigits(imagePath):
    image=cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image Not Found")
        return []

    # Blur
    blurredImage=cv2.GaussianBlur(image,(5,5),0)

    # Binary threshold
    _,thresholdImage=cv2.threshold(blurredImage,120,255,cv2.THRESH_BINARY_INV)

    # Morphological cleanup
    kernel=np.ones((3,3),np.uint8)
    thresholdImage=cv2.morphologyEx(thresholdImage,cv2.MORPH_CLOSE,kernel)

    # Finding contours
    contoursFound,_=cv2.findContours(thresholdImage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    digitImages=[]
    xPositions=[]

    # Minimum contour area
    minimumArea=80
    for singleContour in contoursFound:
        contourArea=cv2.contourArea(singleContour)
        if contourArea<minimumArea:
            continue
        xPos,yPos,boxWidth,boxHeight=(cv2.boundingRect(singleContour))

        # Ignore tiny noises
        if boxWidth<5 or boxHeight<15:
            continue

        # Ignore giant noise
        if boxWidth>image.shape[1]*0.9:
            continue
        extractedDigit=thresholdImage[yPos:yPos+boxHeight,xPos:xPos+boxWidth]

        # Padding
        extractedDigit=cv2.copyMakeBorder(extractedDigit,8,8,8,8,cv2.BORDER_CONSTANT,value=0)
        processedDigit=prepareDigit(extractedDigit)
        digitImages.append(processedDigit)
        xPositions.append(xPos)

    # Sorting left-to-right
    orderedPairs=sorted(zip(xPositions,digitImages),key=lambda item:item[0])
    orderedDigits=[digit for _,digit in orderedPairs]
    return orderedDigits


# Check For Roman Numeral

# In[9]:


# Checking Roman Numeral
def isRomanNumeral(text):
    allowedCharacters=set(
        "IVXLCDMivxlcdm"
    )
    if len(text)==0:
        return False
    return all(
        character in allowedCharacters
        for character in text
    )


# Convert To Roman Numerals

# In[10]:


# Roman Numeral To Integer
def romanToInteger(romanText):
    romanText=romanText.upper()
    romanDictionary={
        'I':1,
        'V':5,
        'X':10,
        'L':50,
        'C':100,
        'D':500,
        'M':1000
    }
    totalValue=0
    previousValue=0
    for character in reversed(romanText):
        currentValue=romanDictionary.get(character,0)
        if currentValue<previousValue:
            totalValue-=currentValue
        else:
            totalValue+=currentValue
        previousValue=currentValue
    return totalValue


# Prediction Script

# - Loads the trained deep learning model and predicts the handwritten digits.
# - Uses segmented digit images to detect multiple numbers in sequence.
# - Displays the final recognized number as output.

# In[11]:


# MAIN PREDICTION FUNCTION
def predictDigits(imagePath):

    # OCR DETECTION
    extractedText=performOCR(imagePath)

    # print("\nOCR Detected :",extractedText)

    # ROMAN NUMERAL CHECK
    if (extractedText is not None and isRomanNumeral(extractedText)):
        convertedNumber=romanToInteger(extractedText)
        print(
            f"\nRoman Numeral "
            f"{extractedText} "
            f"= {convertedNumber}"
        )
        return convertedNumber

    # DIGIT EXTRACTION
    digitList=extractDigits(imagePath)

    if len(digitList)==0:
        print("No Digits Were Found")
        return None
    finalResult=[]
    for indexNumber,singleDigit in enumerate(digitList):
        singleDigit=singleDigit.reshape(1,28,28,1)
        predictionResult=loadedModel.predict(singleDigit,verbose=0)
        detectedNumber=np.argmax(predictionResult)
        confidenceScore=(np.max(predictionResult)*100)
        print(
            f"Digit {indexNumber+1} "
            f"Detected As "
            f"{detectedNumber} "
            f"With "
            f"{confidenceScore:.2f}% "
            f"Confidence"
        )
        finalResult.append(str(detectedNumber))
    joinedNumber="".join(finalResult)
    print("\nFinal Detected Number :",joinedNumber)
    return joinedNumber

# Training Accuracy
plt.figure(figsize=(8,5))
plt.plot(trainingHistory.history['accuracy'],label='Training Accuracy')
plt.plot(trainingHistory.history['val_accuracy'],label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# Loss Graph Of The Model

# Training Loss
plt.figure(figsize=(8,5))
plt.plot(trainingHistory.history['loss'],label='Training Loss')
plt.plot(trainingHistory.history['val_loss'],label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# Confusion Matrix Cell Of The Model

# In[15]:


# Predictions on test data
predictions=digitModel.predict(testImages)
predictedLabels=np.argmax(predictions,axis=1)

# Confusion matrix
confusionMatrix=confusion_matrix(testLabels,predictedLabels)

# Plotting
plt.figure(figsize=(10,8))
sns.heatmap(confusionMatrix,annot=True,
fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Visuals Prediction Cell In Accordance To The Model

# In[16]:


# Showing sample predictions

plt.figure(figsize=(12,10))
for index in range(16):
    plt.subplot(4,4,index+1)
    plt.imshow(testImages[index].reshape(28,28),cmap='gray')
    prediction=np.argmax(
        digitModel.predict(
            testImages[index].reshape(1,28,28,1),
            verbose=0
        )
    )
    plt.title(f"Predicted : {prediction}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# Confidence Visualization Of The Model

# In[17]:


# Predicting one sample

sampleImage=testImages[0]
prediction=digitModel.predict(
    sampleImage.reshape(1,28,28,1),
    verbose=0
)

# Plotting confidence
plt.figure(figsize=(8,5))
plt.bar(range(10),prediction[0])
plt.title("Confidence Score For Each Digit")
plt.xlabel("Digits")
plt.ylabel("Confidence")
plt.xticks(range(10))
plt.show()


# Final Model Evaluation Cell

# In[18]:


# Final evaluation

testLoss,testAccuracy=digitModel.evaluate(
    testImages,
    testLabels,
    verbose=0
)
print(f"Final Test Accuracy : {testAccuracy*100:.2f}%")
print(f"Final Test Loss : {testLoss:.4f}")


# Web Application Support

# - Acts as the backend server for the web application using Flask.
# - Handles image upload functionality from the user interface.
# - Sends uploaded images for prediction and displays the detected digits.

# In[19]:


# FLASK APP
app=Flask(__name__)
uploadFolder="Static/Uploads"
app.config['UPLOAD_FOLDER']=uploadFolder

# Creating upload folder if not exists
os.makedirs(uploadFolder,exist_ok=True)

# HOME PAGE
@app.route("/",methods=["GET","POST"])

def homePage():
    detectedOutput=None
    uploadedImagePath=None
    detectedType=None
    ocrDetectedText=None
    if request.method=="POST":
        uploadedFile=request.files["file"]

        # No file selected
        if uploadedFile.filename=="":
            return "No File Selected"

        if uploadedFile:

            # Save image
            savedPath=os.path.join(app.config['UPLOAD_FOLDER'],uploadedFile.filename)
            uploadedFile.save(savedPath)

            # Fix Windows path
            uploadedImagePath=uploadedFile.filename

            # OCR CHECK
            extractedText=performOCR(savedPath)
            ocrDetectedText=extractedText

            # ROMAN NUMERAL DETECTION
            if (extractedText is not None and isRomanNumeral(extractedText)):
                detectedOutput=romanToInteger(extractedText)
                detectedType="Roman Numeral"

            # HANDWRITTEN DIGIT DETECTION
            else:
                detectedOutput=predictDigits(savedPath)
                detectedType=("Handwritten Digits")

    return render_template(
        "index.html",
        prediction=detectedOutput,
        image_path=uploadedImagePath,
        detected_type=detectedType
    )

# RUN APPLICATION
if __name__=="__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )


# Developed By- Amulya Shrivastava
# LinkedIn- https://www.linkedin.com/in/amulya-shrivastava-11a0a9288/

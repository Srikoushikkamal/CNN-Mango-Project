from pywebio.input import * #for local web
from pywebio.output import *
import numpy as np #for array works
from keras.preprocessing import image #
import cv2
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator 
from functools import partial

#imagedatagenerator is used for increase ability ofthe model
train_datagen = ImageDataGenerator(rescale = 1./255, #Taking count of every pixel
                                   shear_range = 0.2, #To change angles depending of pictures
                                   zoom_range = 0.2, #Random zooming of pics.
                                   horizontal_flip = True) #Mirror image

#Accessing images for training the model                                   
training_set = train_datagen.flow_from_directory('/home/skk/Desktop/Mangoes/train', 
                                                 target_size = (64, 64),#keras is used to resize image
                                                 batch_size = 32, #it's like a for loop to verify each sample image
                                                 class_mode = 'binary') #Returns binary 1 and 0.

'''test data to validate the trained model'''
test_datagen = ImageDataGenerator(rescale = 1./255) #It helps to perform real time along with training images.

#Redefining test data
test_set = test_datagen.flow_from_directory('/home/skk/Desktop/Mangoes/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
print("Image Processing.......Compleated")

#Features of each sample gets saved sequentially
cnn = tf.keras.models.Sequential() #To not rush, and follow a sequence at a time.
print("Building Neural Network.....")

#ADD LAYERS TO YOUR MODEL

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) #Input shape is as-per size and RGB #Convolution Layer applies filters (with Relu as activation function)
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #Another layer to get the features of image
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) # Again appying to classify its features for better results
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) 
cnn.add(tf.keras.layers.Flatten()) #Gets the result in one single pattern
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) #Efficiency layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #Returns values Binary.
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print(" ______ _____Training CNN______ ______")

cnn.fit(x = training_set, validation_data = test_set, epochs = 25) #limits training within the system configuration

check=input("Type Predict : ")
if(check=="Predict" or check=="predict"):
    def start():
        vid = cv2.VideoCapture(0)
        print("Camera Connection Successfully Established")
        i = 0
        pre_out=""
        out1=""
        while(True):
            r, frame = vid.read() 
            cv2.imshow('Ingenious Mango Classifier BETA', frame)
            cv2.imwrite('/home/skk/Desktop/Mangoes/test/final'+str(i)+".jpg", frame)
            test_image = image.load_img('/home/skk/Desktop/Mangoes/test/final'+str(i)+".jpg", target_size = (64, 64))
            test_image = image.img_to_array(test_image) #array patterns
            test_image = np.expand_dims(test_image, axis = 0) #dimensions
            result = cnn.predict(test_image) #limited prediction
            training_set.class_indices
            if result[0][0] ==1:
                out=("Rotten")
            elif result[0][0] == 0:
                out=("Fresh")
            if(out!=pre_out):
                put_text(out)
            if(out1!="Place a Mango"):
                out1=("Place a Mango")
                put_text(out1)
            pre_out=out
            os.remove('/home/skk/Desktop/Mangoes/test/final'+str(i)+".jpg")
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        vid.release()
    def stop():
        put_text("Prediction Stopped")
        put_text("The Updated Mango Is",start(out))
        vid.release()
    put_buttons(['Start Prediction'], onclick=[start])
    put_buttons(['Stop Prediction'], onclick=[stop])
cv2.destroyAllWindows()


#def stop():
#    put_text("Stopped")
#def start():
#    put_text("Started")
#put_buttons(['Start Prediction', 'Stop Prediction'], onclick=[start, stop])

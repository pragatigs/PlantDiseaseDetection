# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.image import imread
# import cv2
# import random
# import os
# from os import listdir
# from PIL import Image
# from sklearn.preprocessing import label_binarize,  LabelBinarizer
# from keras.preprocessing import image
# from tensorflow.keras.utils import img_to_array, array_to_img
# from tensorflow.keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Flatten, Dropout, Dense
# from sklearn.model_selection import train_test_split
# from keras.models import model_from_json
# from tensorflow.keras.utils import to_categorical
# # Plotting 12 images to check dataset
# #Now we will observe some of the iamges that are their in our dataset. We will plot 12 images here using the matplotlib library.
# # plt.figure(figsize=(12,12))
# # path = "D:/R/train2/tt"
# # for i in range(1,17):
# #     plt.subplot(4,4,i)
# #     plt.tight_layout()
# #     rand_img = imread(path +'/'+ random.choice(sorted(os.listdir(path))))
# #     plt.imshow(rand_img)
# #     plt.xlabel(rand_img.shape[1], fontsize = 10)#width of image
# #     plt.ylabel(rand_img.shape[0], fontsize = 10)#height of image
#     #Converting Images to array 
# def convert_image_to_array(image_dir):
#     try:
#         image = cv2.imread(image_dir)
#         if image is not None :
#             image = cv2.resize(image, (256,256))  
#             #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
#             return img_to_array(image)
#         else :
#             return np.array([])
#     except Exception as e:
#         print(f"Error : {e}")
#         return None
# dir = "D:/R/sensor2"
# root_dir = listdir(dir)
# image_list, label_list = [], []
# all_labels = ['Common_rust', 'Early blight', 'Bacterial_spot']
# binary_labels = [0,1,2]
# temp = -1

# # Reading and converting image to numpy array
# #Now we will convert all the images into numpy array.

# for directory in root_dir:
#   plant_image_list = listdir(f"{dir}/{directory}")
#   temp += 1
#   for files in plant_image_list:
#     image_path = f"{dir}/{directory}/{files}"
#     image_list.append(convert_image_to_array(image_path))
#     label_list.append(binary_labels[temp])
# label_counts = pd.DataFrame(label_list).value_counts()
# label_counts.head()

# #Next we will observe the shape of the image.
# image_list[0].shape
# label_list = np.array(label_list)
# label_list.shape
# x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 10) 
# x_train = np.array(x_train, dtype=np.float16) / 225.0
# x_test = np.array(x_test, dtype=np.float16) / 225.0
# x_train = x_train.reshape( -1, 256,256,3)
# x_test = x_test.reshape( -1, 256,256,3)
# y_train = to_categorical(y_train,num_classes=3)
# y_test = to_categorical(y_test,num_classes=3)
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding="same",input_shape=(256,256,3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(8, activation="relu"))
# model.add(Dense(3, activation="softmax"))
# model.summary()
# model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001),metrics=['accuracy'])
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)
# epochs = 20
# batch_size = 128
# history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 
#                     validation_data = (x_val, y_val))
# # plt.figure(figsize=(12, 5))
# # plt.plot(history.history['accuracy'], color='r')
# # plt.plot(history.history['val_accuracy'], color='b')
# # plt.title('Model Accuracy')
# # plt.ylabel('Accuracy')
# # plt.xlabel('Epochs')
# # plt.legend(['train', 'val'])

# # plt.show()
# # print("[INFO] Calculating model accuracy")
# # scores = model.evaluate(x_test, y_test)
# # print(f"Test Accuracy: {scores[1]*100}")
# y_pred = model.predict(x_test)
# img = array_to_img(x_test[0])
# img
# print("Originally : ",all_labels[np.argmax(y_test[0])])
# print("Predicted : ",all_labels[np.argmax(y_pred[0])])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
img_height=224
img_width=224
batch_size=32
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\R\Train',
    labels='inferred',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\R\Train',
    labels='inferred',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# Create the model
input_shape = (224,224,3)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(Flatten())
model.add(Dense(8, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
# Train the model
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\R\Train',
    labels='inferred',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\R\Train',
    labels='inferred',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_data, validation_data=validation_data, epochs=30)
model.save('D:/R')
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\R\Test',
    labels='inferred',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_loss, test_acc = model.evaluate(test_data)

def convert_image_to_array(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            return np.expand_dims(image, axis=0)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None

# Path to the image you want to detect the disease on
image_path = "D:\R\sensor\img\photo_6093712149915481889_m.jpg"  # Replace with the path to your image file

# Convert the image to array
image_array = convert_image_to_array(image_path)

if image_array is not None:
    # Normalize the image array
    image_array = image_array.astype(np.float16) / 225.0

    # Make a prediction
    disease_label = model.predict(image_array)
    predicted_label = np.argmax(disease_label)

    # Define the class labels
    all_labels = ['Common rust', 'Early blight', 'Bacterial spot']

    # Display the predicted disease label
    print("Predicted Disease: ", all_labels[predicted_label])
else:
    print("Failed to load the image.")

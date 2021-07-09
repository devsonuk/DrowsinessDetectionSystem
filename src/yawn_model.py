# Required Imports
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.mobilenet_v3 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


my_data_dir = '/content/drive/MyDrive/data/yawn_data'

# CONFIRM THAT THIS REPORTS BACK 'test', and 'train'
print(os.listdir(my_data_dir))
print(os.listdir(my_data_dir + '/no_yawn')[0])

img = plt.imread(my_data_dir + '/no_yawn/47.jpeg')
plt.imshow(img)

print(img.shape)
print(len(os.listdir(my_data_dir + '/no_yawn')))
print(len(os.listdir(my_data_dir + '/yawn')))

image_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # rotation_range=20, # rotate the image 20 degrees
    # width_shift_range=0.10, # Shift the pic width by a max of 5%
    # height_shift_range=0.10, # Shift the pic height by a max of 5%
    # rescale=1/255, # Rescale the image by normalzing it.
    # shear_range=0.1, # Shear means cutting away part of the image (max 10%)
    # zoom_range=0.1, # Zoom in by 10% max
    # horizontal_flip=True, # Allo horizontal flipping
    # fill_mode='nearest' # Fill in missing pixels with the nearest filled value
)

plt.imshow(img)

plt.imshow(image_gen.random_transform(img))

print(image_gen.flow_from_directory(my_data_dir))

# Data Preprocessing
image_shape = (160, 160, 3)
batch_size = 8

train_image_gen = image_gen.flow_from_directory(my_data_dir,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(my_data_dir,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary', shuffle=False)

print(test_image_gen.class_indices)

# Model Building
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                               include_top=False,
                                               weights='imagenet')
# Model Summary
base_model.trainable = False
print(base_model.summary())

# Flatting the entire neuron result into single dimension
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# Adding the prediction layer
prediction_layer = keras.layers.Dense(1)

# Model construction
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Model Training definition
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
# Final model Summary
print(model.summary())
# Model Training
results = model.fit(train_image_gen,
                    epochs=50,
                    validation_data=test_image_gen
                    )
# Model Save
model.save('../models/yawn_detector.h5')

# Model Prediction
pred = model.predict_classes(test_image_gen)
# Index of data
indexes = test_image_gen.class_indices
print(indexes)

# Classification report
print(classification_report(test_image_gen.classes, pred))

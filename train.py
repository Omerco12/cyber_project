import os.path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

resnet_weights_path = 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
train_dir = os.path.join('dataset', 'train')
test_dir = os.path.join('dataset', 'test')
model_dir = "./models"
# model_name = "/MobileNetV2.h5"


# Specify paths to the training and validation dataset directories
BATCH_SIZE = 32
TARGET_SIZE = (160, 160)

# Define the augmentation parameters for the training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # rescale the pixel values to [0,1]
    rotation_range=20,  # randomly rotate images by up to 20 degrees
    width_shift_range=0.1,  # randomly shift images horizontally by up to 10%
    height_shift_range=0.1,  # randomly shift images vertically by up to 10%
    shear_range=0.2,  # randomly apply shearing transformations
    zoom_range=0.2,  # randomly zoom in on images
    horizontal_flip=True,  # randomly flip images horizontally
    fill_mode='nearest'  # fill any gaps created by the above transformations with the nearest pixel
)

# Define the augmentation parameters for the test data
test_datagen = ImageDataGenerator(
    rescale=1. / 255  # rescale the pixel values to [0,1]
)

# Load the train and test data using flow_from_directory
train_data_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

validation_data_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
sample_training_images, labels = next(validation_data_generator)

nb_train_samples = train_data_generator.samples
nb_validation_samples = validation_data_generator.samples
print(nb_train_samples, nb_validation_samples)

class_labels = train_data_generator.class_indices
print(class_labels)

fp = open('class_labels.json', 'w')
json.dump(class_labels, fp)
fp.close()

base_model = VGG16(input_shape=(160, 160, 3),weights='imagenet', include_top=False)
base_model.summary()


for layer in base_model.layers:
  layer.trainable = False
  print('Layer ' + layer.name + ' frozen.')
# Define early stopping and model checkpoint for optimizing epoch number and saving the best model
# We take the last layer of our the model and add it to our classifier
last = base_model.layers[-1].output
x = Flatten()(last)
#x = Dense(1000, activation='relu', name='fc1')(x)
#x = Dropout(0.3)(x)
prediction = Dense(len(class_labels), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)


optimizer = Adam(learning_rate=0.001)
loss = 'categorical_crossentropy'
metrics = ['accuracy']

# Compile and fit your model
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

epochs = 50

model_checkpoint = ModelCheckpoint(
    filepath='model.h5',
    monitor='accuracy',
    mode='max',
    verbose=1,
    save_best_only=True,
)

early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=3,
    verbose=1,
)
history = model.fit(train_data_generator,
                    steps_per_epoch=len(train_data_generator),
                    epochs=epochs,
                    validation_data=validation_data_generator,
                    validation_steps=len(validation_data_generator),
                    callbacks=[early_stopping, model_checkpoint])
model.save('model.h5')

# Get training and validation accuracy and loss from the history object
# Plot accuracy and loss for testing and validation

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy Value')
plt.xlabel('Epoch')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.title('Training and Validation Loss')
plt.show()

# Load the best saved model
model = load_model('model.h5')

from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(validation_data_generator, nb_validation_samples // BATCH_SIZE + 1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix \n')
print(confusion_matrix(validation_data_generator.classes, y_pred))

print('\n')
print('Classification Report \n')
target_names = class_labels.keys()
print(classification_report(validation_data_generator.classes, y_pred, target_names=target_names))

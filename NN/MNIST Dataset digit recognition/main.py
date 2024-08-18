import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Normalize the data
X_train = X_train / 255.0
X_test = X_test /255.0

#Reshape the data to add a channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1 ,28, 28, 1)

print(f'Training data shape: {X_train.shape}')
print(f'Test data shape: {X_test.shape}')

#Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Summary of model
model.summary()

#Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#Evaluating model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest Accuracy: {test_acc}')

#Confusion Matrix
# Predict the labels for test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
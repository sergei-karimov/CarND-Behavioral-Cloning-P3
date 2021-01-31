import cv2
import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    image_path = line[0]
    image_file_name = image_path.split('\\')[-1]
    current_file_name = f'data/IMG/{image_file_name}'
    image = cv2.imread(current_file_name)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = keras.Sequential()
model.add(layers.Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
print(model.summary())
model.add(layers.Lambda(lambda x: x / 127.5 - 1.0))
print(model.summary())
model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
print(model.summary())
model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
print(model.summary())
model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, epochs=7, shuffle=True)
model.save('model.h5')

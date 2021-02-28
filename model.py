import cv2
import csv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Conv2D, MaxPooling2D, Reshape, Convolution2D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

EPOCH = 50
TEST_SIZE = 0.2
BATCH_SIZE = 32

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


def data_generator(samples, batch_size, augmentation_is_needed=True):
    sample_quantity = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, sample_quantity, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []

            for sample in batch_samples:
                center_image_path = sample[0]
                center_image_file_name = center_image_path.split('/')[-1]
                center_current_file_name = f'data/IMG/{center_image_file_name}'
                center_image = cv2.imread(center_current_file_name)
                images.append(center_image)
                center_measurement = float(line[3])
                angles.append(center_measurement)
                if augmentation_is_needed:
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_measurement * -1.0)

                left_image_path = sample[1]
                left_image_file_name = left_image_path.split('/')[-1]
                left_current_file_name = f'data/IMG/{left_image_file_name}'
                left_image = cv2.imread(left_current_file_name)
                images.append(left_image)
                left_measurement = float(line[3]) + 0.2
                angles.append(left_measurement)
                if augmentation_is_needed:
                    images.append(cv2.flip(left_image, 1))
                    angles.append(left_measurement * -1.0)

                right_image_path = sample[2]
                right_image_file_name = right_image_path.split('/')[-1]
                right_current_file_name = f'data/IMG/{right_image_file_name}'
                right_image = cv2.imread(right_current_file_name)
                images.append(right_image)
                right_measurement = float(line[3]) - 0.2
                angles.append(right_measurement)
                if augmentation_is_needed:
                    images.append(cv2.flip(right_image, 1))
                    angles.append(right_measurement * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)

            X_train, y_train = shuffle(X_train, y_train)

            yield X_train, y_train


def nv_model():
    m = Sequential()
    m.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    m.add(Lambda(lambda x: (x / 255.0) - 0.5))
    m.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    m.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    m.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    m.add(Conv2D(64, (3, 3), activation='relu'))
    m.add(Conv2D(64, (3, 3), activation='relu'))
    m.add(Flatten())
    m.add(Dense(1164, activation="relu"))
    m.add(Dropout(0.2))
    m.add(Dense(100, activation='relu'))
    m.add(Dropout(0.3))
    m.add(Dense(50, activation='relu'))
    m.add(Dropout(0.3))
    m.add(Dense(10, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return m


model = nv_model()
model.summary()

X_train, X_validation = train_test_split(lines, test_size=TEST_SIZE)
train_generator = data_generator(X_train, batch_size=BATCH_SIZE, augmentation_is_needed=True)
validation_generator = data_generator(X_validation, batch_size=BATCH_SIZE, augmentation_is_needed=False)

history_object = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCH,
    validation_steps=len(X_validation) // BATCH_SIZE,
    verbose=1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(
            "model.h5", save_best_only=True, monitor="val_loss", mode="min"
        )
    ]
)

model.save('model.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

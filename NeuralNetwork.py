import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models


categories = ["triangle", "square"]

# dane do trenowania
if not os.path.exists("training_images.npy"):
    data_dir = "F:\Gra_vision-voice\images"
    img_size = 400
    training_data = []

    for category in categories:
        path = os.path.join(data_dir, category)
        class_number = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array, class_number])

    random.shuffle(training_data)

    training_images = []
    training_labels = []
    for img, label in training_data:
        training_images.append(img)
        training_labels.append(label)
    training_images = np.array(training_images).reshape(-1, img_size, img_size, 1)
    training_labels = np.array(training_labels)

    np.save("training_images", training_images)
    np.save("training_labels", training_labels)

# uczenie


training_images = np.load("training_images.npy")
training_labels = np.load("training_labels.npy")

training_images = training_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=training_images.shape[1:]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Dense(len(categories), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
num_classes = len(categories)
training_labels = keras.utils.to_categorical(training_labels, num_classes=num_classes)
model.fit(training_images, training_labels, epochs=10, validation_split=0.1, batch_size=32)

model.save('shape_classifier.model')




# testowanie
model = models.load_model('shape_classifier.model')
for img in os.listdir("F:\Gra_vision-voice\images\square"):
    img_test = cv2.imread(os.path.join("F:\Gra_vision-voice\images\square", img), cv2.IMREAD_GRAYSCALE)
    new_img_test = cv2.resize(img_test, (400, 400))
    prediction = model.predict(np.array([new_img_test]) / 255)
    index = np.argmax(prediction)
    print(categories[index])
    plt.imshow(new_img_test, cmap='gray')
    plt.show()


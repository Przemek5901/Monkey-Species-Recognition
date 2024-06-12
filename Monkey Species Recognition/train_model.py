import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models

warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Define paths for training and testing data
train_data_dir = 'D:\\Monkey-Species-Recognition\\Monkey Species Recognition\\dataset\\training'
test_data_dir = 'D:\\Monkey-Species-Recognition\\Monkey Species Recognition\\dataset\\validation'

# Class mapping
class_mapping = {
    0: 'mantled howler',
    1: 'patas monkey',
    2: 'bald uakari',
    3: 'japanese macaque',
    4: 'pygmy marmoset',
    5: 'white headed_capuchin',
    6: 'silvery marmoset',
    7: 'common squirrel_monkey',
    8: 'black headed night monkey',
    9: 'nilgiri langur',
}

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Split data into training and validation sets
target_size = (128, 128)

train_dataset = train_datagen.flow_from_directory(train_data_dir,
                                                  subset="training",
                                                  seed=123,
                                                  target_size=target_size,
                                                  batch_size=32,
                                                  class_mode='sparse')

validation_dataset = train_datagen.flow_from_directory(train_data_dir,
                                                       subset="validation",
                                                       seed=123,
                                                       target_size=target_size,
                                                       batch_size=32,
                                                       class_mode='sparse')

test_dataset = test_datagen.flow_from_directory(test_data_dir,
                                                target_size=target_size,
                                                batch_size=32,
                                                class_mode='sparse')

# Build the model
num_classes = 10

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),  # Adding Dropout layer
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Adding Dropout layer
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 1
history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=epochs)

# Evaluate the model
loss, accuracy = model.evaluate(validation_dataset)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the model in .keras format
model.save('monkey_species_recognition_model.keras')

# Prediction on test data
for images, labels in test_dataset:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"Actual classes: {labels}")
    print(f"Predicted classes: {predicted_classes}")
    break

# Prediction on a single image
img_path = 'D:\\Monkey-Species-Recognition\\Monkey Species Recognition\\dataset\\validation\\n0\\n0010.jpg'
try:
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    print(f"Predicted class for the single image: {class_mapping[predicted_class]}")

except FileNotFoundError:
    print(f"The file {img_path} was not found. Check the path.")

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Define paths for testing data
test_data_dir = 'D:\\Monkey-Species-Recognition\\Monkey Species Recognition\\dataset\\validation'

# Class mapping
class_mapping = {
    0: 'mantled_howler',
    1: 'patas_monkey',
    2: 'bald_uakari',
    3: 'japanese_macaque',
    4: 'pygmy_marmoset',
    5: 'white_headed_capuchin',
    6: 'silvery_marmoset',
    7: 'common_squirrel_monkey',
    8: 'black_headed_night_monkey',
    9: 'nilgiri_langur',
}

# Data augmentation for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Split data into test sets
target_size = (128, 128)
test_dataset = test_datagen.flow_from_directory(test_data_dir,
                                                target_size=target_size,
                                                batch_size=32,
                                                class_mode='sparse',
                                                shuffle=False)

# Load the trained model
model_path = 'monkey_species_recognition_model.keras'
model = load_model(model_path)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Prediction on test data
for images, labels in test_dataset:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"Actual classes: {labels}")
    print(f"Predicted classes: {predicted_classes}")
    break  # Remove this line if you want to test more batches


# Getting Started with Image Classification: Horses vs. Humans

This guide will help you set up a deep learning model to classify images of horses and humans. The model uses Convolutional Neural Networks (CNNs) in TensorFlow.

---

## Install Required Libraries
To run this project, you need to install TensorFlow, NumPy, and Matplotlib. If you are using Google Colab, run the following command (using `!` before `pip`):

```python
!pip install tensorflow numpy matplotlib
```

For other environments, just run:
```bash
pip install tensorflow numpy matplotlib
```

This installs TensorFlow (for building and training the deep learning model), NumPy (for numerical operations), and Matplotlib (for visualization purposes).

---

## Download the Dataset
We will download the **Horses or Humans dataset** using KaggleHub.

```python
import kagglehub

path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")

print("Path to dataset files:", path)
```

This fetches the dataset from Kaggle, which contains labeled images of horses and humans. The downloaded files are stored in the specified path, which will be displayed in the output.

---

## Prepare the Data

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = path + '/horses-or-humans/train'

train_datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)
```

### What Happens Here?
1. **ImageDataGenerator** helps preprocess the images by normalizing pixel values (0 to 1) for better model performance.
2. **flow_from_directory** automatically loads images from the dataset directory, resizes them to 300x300 pixels, and prepares batches of 32 images.
3. **class_mode='binary'** ensures labels are assigned as 0 or 1, making it a binary classification task.

---

## Build the CNN Model

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Understanding the Model:
1. **Conv2D Layers** - These layers detect patterns in images (like edges, shapes, and textures).
2. **MaxPooling2D Layers** - These reduce the size of the image representation, keeping important features while removing unnecessary details.
3. **Flatten Layer** - Converts the 2D features extracted by convolutional layers into a 1D array.
4. **Dense Layers** - Fully connected layers process extracted features and make the final classification.
5. **Sigmoid Activation** - Outputs a probability score (closer to 1 = human, closer to 0 = horse).

---

## Compile and Train the Model

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=15
)
```

### What Happens During Training?
1. **Binary Cross-Entropy Loss** - Measures how well the model predicts between two categories.
2. **RMSprop Optimizer** - Adjusts model weights to minimize the loss function and improve accuracy.
3. **Training for 15 Epochs** - The model processes all images 15 times to learn patterns effectively.
4. **Accuracy Metric** - Helps track how well the model is classifying images during training.

---

## Make Predictions

```python
import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        print("Predicted: Human")
    else:
        print("Predicted: Horse")

image_path = "/content/humanimage.jpg"
predict_image(image_path)
```

### How Does This Work?
1. **Load an image** - The function takes an image file as input and resizes it to 300x300 pixels.
2. **Convert to NumPy Array** - The image is converted into an array format that the model can process.
3. **Normalize Pixel Values** - The image data is scaled between 0 and 1.
4. **Make a Prediction** - The trained model analyzes the image and classifies it as either a horse or a human.
5. **Print the Output** - Based on the prediction probability, the image is classified accordingly.

---

## Summary
1. **Install TensorFlow & other libraries.**
2. **Download the dataset using KaggleHub.**
3. **Preprocess images using ImageDataGenerator.**
4. **Build a CNN model using TensorFlow.**
5. **Train the model on the dataset.**
6. **Test the model on new images.**

---

## Additional Resources
- **MaxPooling Explanation:** [MaxPooling Guide](https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/content/images/2022/07/maxpooled_1-1.png)
- **Building a Keras Model:** [Keras Model Guide](https://makeschool.org/mediabook/oa/tutorials/keras-for-image-classification-pfw/building-a-keras-sequential-model/)

This guide should be easy to follow, even for absolute beginners. 🚀


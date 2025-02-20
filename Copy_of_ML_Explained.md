Make sure you have the following installed:
```
!pip install tensorflow numpy matplotlib
```

Note:-use (!) only in colab Notebook
### Code Explanation:
This code downloads the 'Horses or Humans' dataset from KaggleHub, which contains images of horses and humans used for classification tasks and save and Display's the path address.

```python
import kagglehub

path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")

print("Path to dataset files:", path)
```

---

### Code Explanation:
This code sets up the image preprocessing pipeline using TensorFlow's `ImageDataGenerator`. It rescales pixel values and loads images from the specified training directory.

train_dir:paste the path of the dataset directory(already pasted in this example)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

train_dir = path+'/horses-or-humans/train'
```
train_dir:paste the path of the dataset directory(already pasted in this example)
        Here we pass the path of our datasets
```
train_datagen = ImageDataGenerator(rescale=1/255.0)
```
ImageDataGenerator(rescale=1/255.0) is used in deep learning (especially with TensorFlow/Keras) to preprocess images.

rescale=1/255.0 scales pixel values from 0-255 to 0-1 (normalization).
This helps the model train better by making the data easier to process.
```

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary')
```
This code loads images from a directory and prepares them for training using ImageDataGenerator. Here's a simple breakdown:

train_dir → Path to the folder containing training images.
target_size=(300, 300) → Resizes all images to 300x300 pixels.
batch_size=32 → Loads 32 images at a time (batch processing).
class_mode='binary' → Labels images as 0 or 1 (for binary classification, like cat vs. dog).
This is useful for training deep learning models with images efficiently! 
---

### Code Explanation:
This code defines a Convolutional Neural Network (CNN) model using Keras. It consists of multiple convolutional and max-pooling layers followed by dense layers for binary classification.

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

---

### Code Explanation:
This code compiles and trains the CNN model using binary cross-entropy loss and the RMSprop optimizer. The model is trained for 15 epochs using the training data generator.

```python
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

---

### Code Explanation:
This code defines helper functions to preprocess input images and make predictions using the trained model. It loads an image, resizes it, normalizes pixel values, and predicts whether it is a horse or a human.

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        print(f"Predicted: Human")
    else:
        print(f"Predicted: Horse")

image_path = "/content/humanimage.jpg"
predict_image(image_path)
```

---

### Additional Resources:
- MaxPooling Explanation: [Link](https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/content/images/2022/07/maxpooled_1-1.png)
- Building a Keras Model: [Link](https://makeschool.org/mediabook/oa/tutorials/keras-for-image-classification-pfw/building-a-keras-sequential-model/)


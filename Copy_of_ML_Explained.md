

### Code Explanation:
This code performs the following operations:

(Add detailed explanation here if available in the markdown cells above or below)


```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")

print("Path to dataset files:", path)
```

### Code Explanation:
This code performs the following operations:

(Add detailed explanation here if available in the markdown cells above or below)


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


train_dir = ''


train_datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary')

```

### Code Explanation:
This code performs the following operations:

(Add detailed explanation here if available in the markdown cells above or below)


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

### Code Explanation:
This code performs the following operations:

(Add detailed explanation here if available in the markdown cells above or below)


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

### Code Explanation:
This code performs the following operations:

(Add detailed explanation here if available in the markdown cells above or below)


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

Maxpooling- https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/content/images/2022/07/maxpooled_1-1.png


building a keras model-https://makeschool.org/mediabook/oa/tutorials/keras-for-image-classification-pfw/building-a-keras-sequential-model/

### Code Explanation:
This code performs the following operations:

(Add detailed explanation here if available in the markdown cells above or below)


```python

```
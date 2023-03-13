import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator as idg
import PIL
import scipy
import numpy as np
from keras.utils import load_img, img_to_array

# PREPROCESSING
train_data_object = idg(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
training_set = train_data_object.flow_from_directory(
    'training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)
test_data_object = idg(rescale = 1./255)
test_set = test_data_object.flow_from_directory(
    'test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

# BUILDING THE CNN
cnn = tf.keras.models.Sequential()  # Initialization
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))  # 1st layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))  # Pooling 1st layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))  # 2nd layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))  # Pooling 2nd layer
cnn.add(tf.keras.layers.Flatten())      # Flattening
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))    # Connecting
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))   # Output Layer

# TRAINING CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)  # SHOULD TAKE ABOUT 15-20 MINUTES DEPENDING ON YOUR SYSTEM

# TESTING
test_image = load_img('prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image/255.0)
ind = training_set.class_indices
print(ind)
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
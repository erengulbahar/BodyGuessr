from tensorflow import keras, expand_dims, nn
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, RandomFlip, RandomRotation, RandomZoom, Dropout
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import get_file, load_img, img_to_array, image_dataset_from_directory
from keras.models import save_model, load_model
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
img_height = 180
img_width = 180

#validation_split and seed using together
train_ds = image_dataset_from_directory("Bodies", validation_split=0.2, subset="training", seed=10, image_size=(img_height, img_width), batch_size=batch_size)
validation_ds = image_dataset_from_directory("Bodies", validation_split=0.2, subset="validation", seed=10, image_size=(img_height, img_width), batch_size=batch_size)

"""train_ds = image_dataset_from_directory("Bodies", image_size=(img_height, img_width), batch_size=batch_size)
validation_ds = image_dataset_from_directory("Bodies", image_size=(img_height, img_width), batch_size=batch_size)"""

class_names = train_ds.class_names

#Show images
"""plt.figure(figsize=(7, 7))

for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()"""

#Show shape of images & labels
"""for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break"""

#We change to colour of photos because we want to be same type all photos
"""normalization_layer = Rescaling(1. /255)
normalizated_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, label_batch = next(iter(normalizated_ds))"""

#We create model and layers
num_classes = len(class_names)

#We reduce overfitting, these two lines equals
#data_augmentation = Sequential([RandomFlip("horizontal", input_shape=(img_height,img_width,3)), RandomRotation(0.1), RandomZoom(0.1)])
"""data_augmentation = Sequential()
data_augmentation.add(RandomFlip("horizontal", input_shape=(img_height, img_width, 3)))
data_augmentation.add(RandomRotation(0.1))
data_augmentation.add(RandomZoom(0.1))

model = Sequential()
model.add(data_augmentation)
model.add(Rescaling(1. /255, input_shape=(img_height, img_width, 3)))
model.add(Conv2D(16, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="softmax"))
model.add(Dense(num_classes))

#We compile model
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

#Show info about layers and params
model.summary()

#We train the model
epochs = 25
history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs)"""

#Save model
#model.save("MyModel2", overwrite=True)

#Show to graphic about model train accuracy and validation accuracy
"""acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()"""

#sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
#sunflower_path = get_file('Red_sunflower', origin=sunflower_url)

#rose_url = "./red-roses.jpg"

#Load model
loaded_model = load_model("MyModel")

"""test_image = "testImage6.jpg"

img = load_img(test_image, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = nn.softmax(predictions[0])

print(score)

print("This image most likely belongs to {} with a %{:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))"""

#Load data from file
test_image = "testImage.jpg"

img = load_img(test_image, target_size=(img_height, img_width))
#cover image to array
img_array = img_to_array(img)
img_array = expand_dims(img_array, 0)

predictions = loaded_model.predict(img_array)
score = nn.softmax(predictions[0])

print(score)

print("This image most likely belongs to {} with a %{:.2f} percent confidence.".format(class_names[np.argmax(score)], 100* np.max(score)))
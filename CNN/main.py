import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and Preprocess the Data
# CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Explore the Data (Optional)
# Uncomment the following lines to visualize the first 25 images in the training set

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # The CIFAR labels are arrays, so you need the first element
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# 3. Build the CNN Model
model = models.Sequential()
# First convolutional layer, 32 filters, 3x3 kernel, ReLU activation
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# First max pooling layer, 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))

# Second convolutional layer, 64 filters
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Second max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Third convolutional layer, 64 filters
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 4. Add Dense Layers on Top
model.add(layers.Flatten())  # Flatten the output to feed into dense layers
model.add(layers.Dense(64, activation='relu'))  # Fully connected layer
model.add(layers.Dense(10))  # Output layer, one for each class

# 5. Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 6. Train the Model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# 7. Evaluate the Model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# 8. Make Predictions (Optional)
# Convert the model's logits to probabilities
probability_model = models.Sequential([model, 
                                       tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Function to plot the image and prediction
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i][0], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{class_names[predicted_label]} ({100*np.max(predictions_array):2.0f}%)",
               color=color)

# Plot the first X test images, their predicted labels, and the true labels
# to visually check the performance of the model
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
plt.tight_layout()
plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Output the shapes of the datasets
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

# Output the number of training and test samples
print("\nNumber of training samples:", x_train.shape[0])
print("Number of test samples:", x_test.shape[0])

# Output the number of unique classes
num_classes = len(np.unique(y_train))
print("\nNumber of classes:", num_classes)

# Output class labels for reference
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("Class names:", class_names)

# Display a few sample images with their labels


def display_sample_images(x_data, y_data, class_names, num_samples=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_data[i])
        plt.xlabel(class_names[int(y_data[i])])
    plt.show()


print("\nSample training images:")
display_sample_images(x_train, y_train, class_names)

print("\nSample test images:")
display_sample_images(x_test, y_test, class_names)

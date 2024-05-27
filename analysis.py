import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


# Load the test data
_, _, x_test, y_test = load_data()

# Load the saved model
model = tf.keras.models.load_model('Model.h5')

# Get the predictions
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate the accuracy
accuracy = (y_pred == y_true).mean()
print(f"Test Accuracy: {accuracy:.4f}")

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=[
                               "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
print(report)

# Plot the confusion matrix
class_names = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
            yticklabels=class_names, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the image
plt.savefig('images/analysis.png')

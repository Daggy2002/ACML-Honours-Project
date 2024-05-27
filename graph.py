import re
import matplotlib.pyplot as plt

# Open the text file and read its contents
with open('node_output/output.txt', 'r') as file:
    lines = file.readlines()

# Define a regular expression to match the desired output line
output_pattern = r'(\d+)/(\d+) \[\=+\] - \d+s \d+ms/step - loss: (\d+\.\d+) - accuracy: (\d+\.\d+) - val_loss: (\d+\.\d+) - val_accuracy: (\d+\.\d+)'

epochs = []
losses = []
accuracies = []
val_losses = []
val_accuracies = []

for line in lines:
    output_match = re.match(output_pattern, line)
    if output_match:
        epoch_num = len(epochs) + 1
        loss = float(output_match.group(3))
        accuracy = float(output_match.group(4))
        val_loss = float(output_match.group(5))
        val_accuracy = float(output_match.group(6))

        epochs.append(epoch_num)
        losses.append(loss)
        accuracies.append(accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

# Plot the loss and accuracy values
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')

plt.subplot(2, 2, 2)
plt.plot(epochs, accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')

plt.subplot(2, 2, 3)
plt.plot(epochs, val_losses)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss over Epochs')

plt.subplot(2, 2, 4)
plt.plot(epochs, val_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy over Epochs')

plt.tight_layout()
plt.show()

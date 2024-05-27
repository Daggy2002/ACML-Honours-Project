import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split


# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU not available, using CPU.")

# Load and preprocess the CIFAR-10 dataset
def load_data():
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x, x_test = x / 255.0, x_test / 255.0
    y, y_test = tf.keras.utils.to_categorical(y, 10), tf.keras.utils.to_categorical(y_test, 10)

    # Split the data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test

# Define the input shape
def get_input_shape(x_train):
    input_shape = x_train.shape[1:]
    return input_shape

# Apply data augmentation
def augment_data(x_train):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)
    return datagen

# Build the model
def build_model(input_shape):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    return model

# Compile the model
def compile_model(model, learning_rate, optimizer):
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
def train_model(model, x_train, y_train, x_val, y_val, datagen, epochs=100, patience=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                        epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=[early_stopping])
    return history
# Evaluate the model
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
    return test_acc

def plot_metrics(history):
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Save the model with a timestamp and accuracy in the filename
def save_model(model, test_acc):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}_acc_{test_acc:.4f}.h5"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")

def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    datagen = augment_data(x_train)
    input_shape = get_input_shape(x_train)
    model = build_model(input_shape)

    learning_rate = 0.01
    optimizer = 'adam'
    epochs = 100
    patience = 10

    compile_model(model, learning_rate, optimizer)
    history = train_model(model, x_train, y_train, x_val, y_val,  datagen, epochs, patience)
    test_acc = evaluate_model(model, x_test, y_test)
    plot_metrics(history)
    save_model(model, test_acc)

if __name__ == "__main__":
    main()

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Image preprocessing function
def preprocess_image(image_path, target_size=(64, 64)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return img_array

# List of all image paths
healthy_leaf_paths = ["healthy_1.jpg", "healthy_2.jpg", "healthy_3.jpg", "healthy_4.jpg"]
diseased_leaf_paths = ["diseased_1.jpg", "diseased_2.jpg", "diseased_3.jpg", "diseased_4.jpg"]

# Create dataset
X = []
y = []

for path in healthy_leaf_paths:
    X.append(preprocess_image(path))
    y.append(0)

for path in diseased_leaf_paths:
    X.append(preprocess_image(path))
    y.append(1)

X = np.array(X)
y = to_categorical(y, 2)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X, y, epochs=10, verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.title('Training Accuracy and Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_plot.png')

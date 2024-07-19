import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# Set the paths and parameters
data_dir = r'C:\Users\Vrishin Dharmesh KP\Downloads\gesture\leapGestRecog'
categories = [str(i).zfill(2) for i in range(10)]  # Assuming categories are labeled as '00', '01', ..., '09'
img_size = 64  # Resize images to 64x64
epochs = 10
batch_size = 32

def load_images(data_dir, categories, img_size):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            for img in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, img)
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    resized_array = cv2.resize(img_array, (img_size, img_size))
                    data.append(resized_array)
                    labels.append(category)
                except Exception as e:
                    pass
    return np.array(data), np.array(labels)

# Load and preprocess the data
data, labels = load_images(data_dir, categories, img_size)
data = data / 255.0  # Normalize images
data = data.reshape(-1, img_size, img_size, 1)  # Reshape for CNN

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output layer for gesture classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print(f"Accuracy: {accuracy_score(y_test_labels, y_pred_labels)}")
print(classification_report(y_test_labels, y_pred_labels, target_names=categories))

# Save the model
model.save('hand_gesture_model.h5')

# Example of using the model for prediction
def predict_gesture(image_path, model, le, img_size):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to read image file: {image_path}")

        img = cv2.resize(img, (img_size, img_size))
        img = img.reshape(1, img_size, img_size, 1) / 255.0
        prediction = model.predict(img)
        predicted_label = le.inverse_transform(np.argmax(prediction, axis=-1))
        return predicted_label[0]
    except Exception as e:
        print(f"Error predicting gesture: {e}")
        return None


# Example usage
example_image_path = r'C:\Users\Vrishin Dharmesh KP\Downloads\gesture\leapGestRecog\07\09_c\frame_07_09_0169.png'
predicted_gesture = predict_gesture(example_image_path, model, le, img_size)
print(f"Predicted gesture: {predicted_gesture}")

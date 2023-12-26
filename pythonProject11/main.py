import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models

# Load an image from file
image = cv2.imread('omr_sheet.jpg')

# Check if the image was loaded successfully
if image is not None:
    # Apply a Gaussian blur filter
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Display the original and filtered images
    cv2.imshow('Original Image', image)
    cv2.imshow('Blurred Image', blurred_image)
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(450, 300, 3)),  # Update input shape
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(5, activation="softmax"),  # Assuming there are 5 classes (choices)
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Load pre-trained weights if available
    # model.load_weights("your_pretrained_weights.h5")

    # Run the preprocessed image through the model
    img_tensor = tf.expand_dims(blurred_image, axis=0)  # Add batch dimension
    img_tensor = tf.image.resize(
        img_tensor, (450, 300)
    )  # Resize to the expected input shape
    predictions = model.predict(img_tensor)
    # Process the predictions as needed
    # ...

    # Save the model
    model.save("your_model.h5")

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Error: Image not loaded.')


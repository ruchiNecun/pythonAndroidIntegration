import cv2
import numpy as np
from tensorflow.keras import layers, models

# Load an image from file
image = cv2.imread("cat.jpeg")
widthImg = 400
heightImg = 400

# Check if the image was loaded successfully
if image is not None:
    image = cv2.resize(image, (widthImg, heightImg))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define a simple neural network
    model = models.Sequential()
    model.add(
        layers.Flatten(input_shape=(heightImg, widthImg))
    )  # Flatten the input image
    model.add(
        layers.Dense(10, activation="relu")
    )  # Dense layer with 128 neurons and ReLU activation
    model.add(
        layers.Dense(heightImg * widthImg, activation="sigmoid")
    )  # Dense layer with output neurons equal to image size and sigmoid activation
    model.add(
        layers.Reshape((heightImg, widthImg))
    )  # Reshape the output to match image dimensions

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model on the grayscale image
    model.fit(
        np.expand_dims(gray_img, axis=0), np.expand_dims(gray_img, axis=0), epochs=10
    )

    # Save the model in .h5 format
    model.save("Network.h5")

    # Display the original and filtered images
    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", gray_img)

    # Apply the trained model to the input image
    predicted_img = model.predict(np.expand_dims(gray_img, axis=0))[0]

    # Display the predicted image

    cv2.waitKey(0)
    cv2.destroyAllWindows()
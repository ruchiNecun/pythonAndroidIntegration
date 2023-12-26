import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("your_model.h5")

# Create the converter and enable quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization

# Convert and save the quantized TFLite model
tflite_model = converter.convert()
with open("quantized_opencv_model1.tflite", "wb") as f:
    f.write(tflite_model)
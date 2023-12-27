import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("Network.h5")

# Create the converter and enable quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization

# Convert and save the quantized TFLite model
tflite_model = converter.convert()
with open("Gray.tflite", "wb") as f:
    f.write(tflite_model)
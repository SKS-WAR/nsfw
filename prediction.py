import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('image_classifier.h5')

# Load the test image
img_path = r'C:\Users\Sudeep Kumar Sahoo\OneDrive\Pictures\Screenshots\Screenshot (27).png'
img = image.load_img(img_path, target_size=(224, 224))

# Preprocess the image
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make predictions
preds = model.predict(img_array)

# Print the predicted class
print(preds)

# Define class names
class_names = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

# Print the predicted class
class_idx = np.argmax(preds)
class_name = class_names[class_idx]
print('Predicted class:', class_name)

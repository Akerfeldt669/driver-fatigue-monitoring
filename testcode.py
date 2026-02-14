import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1️⃣ Load your trained model
model_path = r"C:\Users\...\...\driver-fatigue-monitoring\models\saved\eye_classifier.h5"
model = load_model(model_path)
print("✅ Model loaded successfully")

# 2️⃣ Load an image to test
img_path = r"C:\Users\...\...\data\test\sleepy\s0037_04192_1_1_0_0_0_01.png"
img = image.load_img(
    img_path,
    target_size=(32, 32),
    color_mode="grayscale"  # your model expects grayscale
)

# 3️⃣ Convert image to array and normalize
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# 4️⃣ Make prediction
pred = model.predict(img_array)
print("Raw output:", pred)

# 5️⃣ Interpret prediction
if pred[0][1] > pred[0][0]:
    print("😴 Sleepy")
else:
    print("👀 Awake")

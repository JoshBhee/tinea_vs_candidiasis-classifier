from tensorflow.keras.models import load_model

model = load_model("skin_model.h5")
print("Model loaded successfully!")
model.summary()
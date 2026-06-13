import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("best_model_v2.keras")

# -----------------------------
# VALIDATION DATA
# -----------------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_generator = datagen.flow_from_directory(
    "dataset_binary",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# -----------------------------
# PREDICTIONS
# -----------------------------
pred_probs = model.predict(val_generator)

predictions = (pred_probs > 0.5).astype(int).flatten()

y_true = val_generator.classes

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, predictions)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        predictions,
        target_names=["Benign", "Malignant"]
    )
)

# -----------------------------
# ACCURACY
# -----------------------------
accuracy = np.mean(predictions == y_true)

print(f"\nAccuracy: {accuracy*100:.2f}%")
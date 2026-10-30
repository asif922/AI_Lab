import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Step 1: Input data (Study hours)
X = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8],
    dtype=np.float32
)

# Step 2: Output data
# 0 = Fail, 1 = Pass
y = np.array(
    [0, 0, 0, 0, 1, 1, 1, 1],
    dtype=np.float32
)

# Reshape input data
X = X.reshape(-1, 1)

# Step 3: Create ANN model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 4: Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train ANN
model.fit(X, y, epochs=100, verbose=1)

# Step 6: Test prediction
hours = np.array([[6.0]], dtype=np.float32)

prediction = model.predict(hours, verbose=0)

# Step 7: Display prediction
probability = prediction[0][0]

print("\nProbability:", round(float(probability), 4))

if probability >= 0.5:
    print("Result: PASS")
else:
    print("Result: FAIL")

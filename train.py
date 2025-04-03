from data_preparation import prepare_data
import tensorflow as tf

# Get datasets
train_dataset, val_dataset = prepare_data()

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=20001, output_dim=128),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dense(20001, activation='softmax')
])

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    steps_per_epoch=100,  # Limit training to 100 iterations per epoch
    validation_steps=50,  # Limit validation to 50 iterations per epoch
)

# Save model
model.save('next_word_model.h5')
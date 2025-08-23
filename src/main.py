import Model_Definition as md
import Dataset_Utils as du
import numpy as np
import tensorflow as tf

model = md.ResNet50_UNet()
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)

datasets = du.get_dataset_paths("./dataset")
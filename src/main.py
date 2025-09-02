import Model_Definition as md
import Dataset_Utils as du
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


model = md.ResNet50_UNet()
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)

datasets = du.get_dataset_paths("./dataset")

input_image_paths = datasets[:,0]
target_image_paths = datasets[:,1]

dataset = tf.data.Dataset.from_tensor_slices((input_image_paths, target_image_paths))
dataset = dataset.shuffle(buffer_size=len(input_image_paths))
dataset = dataset.map(du.load_img_pair, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(16)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

checkpoint_cb = ModelCheckpoint(
    "/content/drive/MyDrive/best_model_20250902.keras",        
    monitor="loss",         
    save_best_only=True,    
    save_weights_only=False,
    verbose=1
)

model.fit(dataset, epochs=20, callbacks=[checkpoint_cb])
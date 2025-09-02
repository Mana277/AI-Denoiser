import glob
import os
import numpy as np
import tensorflow as tf

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3


def get_dataset_paths(dataset_dir_path, ext="jpg"):
  noisy_pattern = os.path.join(dataset_dir_path, f"noisy/*.{ext}")  
  noisy_img_paths = glob.glob(noisy_pattern)
  img_paths = []
  for noisy_img_path in noisy_img_paths:
    filename = os.path.basename(noisy_img_path)  
    base = filename.split("_")[0]                 
    clean_pattern = os.path.join(dataset_dir_path, f"clean/{base}.{ext}")  
    if os.path.exists(clean_pattern):
      img_paths.append((noisy_img_path, clean_pattern))
    else:
      print(f"学習データ{noisy_img_path}に対応する正解ラベル{clean_pattern} は存在しません")
  
  if not img_paths:
    print("学習データが存在しません")
  return np.array(img_paths)

def load_img_pair(noisy_path, clean_path):  # Image loading function for ResNet50
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
    
    noisy = tf.io.read_file(noisy_path)                           
    noisy = tf.image.decode_jpeg(noisy, channels=IMG_CHANNELS)                
    noisy = tf.image.resize(noisy, [IMG_WIDTH, IMG_HEIGHT])       
    noisy = tf.cast(noisy, tf.float32)                            
    noisy = noisy[..., ::-1]                                       
    noisy = noisy - mean                                           

    clean = tf.io.read_file(clean_path)
    clean = tf.image.decode_jpeg(clean, channels=IMG_CHANNELS)
    clean = tf.image.resize(clean, [IMG_WIDTH, IMG_HEIGHT])
    clean = tf.cast(clean, tf.float32)
    clean = clean[..., ::-1]                                       
    clean = clean - mean

    return noisy, clean

def rescale_to_uint8_img(img): #Rescales a ResNet50-normalized image to [0, 255] values.
    img[:, :, :, 0] += 103.939
    img[:, :, :, 1] += 116.779   
    img[:, :, :, 2] += 123.68

    img = img[:, :, :, ::-1]

    rescale_img = np.clip(img, 0, 255).astype('uint8')
    return rescale_img
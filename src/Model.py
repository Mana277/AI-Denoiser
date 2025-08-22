import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3

def load_img(image_path):  # Image loading function for ResNet50
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    img = image.load_img(image_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    return preprocessed_img

def rescale_to_uint8_img(img): #Rescales a ResNet50-normalized image to [0, 255] values.
    img[:, :, :, 0] += 103.939
    img[:, :, :, 1] += 116.779   
    img[:, :, :, 2] += 123.68

    img = img[:, :, :, ::-1]

    rescale_img = np.clip(img, 0, 255).astype('uint8')
    return rescale_img

def ResNet50_UNet(): #Model Definition
    encoder = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    encoder.trainable = False

    # Names of layers to extract skip connection features
    skip_connection_names = [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
    ]

    skip_connections = [encoder.get_layer(name).output for name in skip_connection_names]
    bottleneck = encoder.output

    def decoder_block(input_tensor, skip_tensor):
        filters = skip_tensor.shape[3]
        x = Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(input_tensor)
        x = Concatenate()([x, skip_tensor])
        x = Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = Conv2D(filters, 3, padding='same', activation='relu')(x)
        return x

    d1 = decoder_block(bottleneck, skip_connections[3]) 
    d2 = decoder_block(d1, skip_connections[2])          
    d3 = decoder_block(d2, skip_connections[1])
    d4 = decoder_block(d3, skip_connections[0])

    final_upsample = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(d4)
    outputs = Conv2D(3, 3, activation=None, padding='same')(final_upsample)

    model = Model(inputs=encoder.input, outputs=outputs)
    return model


model = ResNet50_UNet()
model.summary()
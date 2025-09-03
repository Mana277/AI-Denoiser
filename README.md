# AI-Denoiser
An AI-powered tool to automatically remove noise from images, enhancing clarity and quality using deep learning models. 
## Features 
- Uses a **ResNet50-based U-Net architecture** for image denoising.
- Supports color images with arbitrary sizes. - Pretrained encoder weights (ImageNet) for better feature extraction.
- Lightweight decoder for efficient inference.
## Model Architecture The AI-Denoiser uses a **ResNet50 U-Net** model:
- **Encoder:**
   Pretrained ResNet50 (ImageNet) with frozen weights.
- **Skip connections:**
  Extract features from multiple layers (conv1_relu, conv2_block3_out, conv3_block4_out, conv4_block6_out) for high-resolution reconstruction.
- **Decoder:**
  Transposed convolutions and concatenation with skip connections, followed by convolution layers with ReLU activations.
- **Output:**
  Denoised 3-channel image (RGB).

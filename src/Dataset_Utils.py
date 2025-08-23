import glob
import os
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
  return img_paths


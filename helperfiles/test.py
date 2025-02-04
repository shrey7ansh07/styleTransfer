# from PIL import Image
# import os
# import random
# import numpy as np

# directory_colored = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/dataset/colored"
# directory_black_and_white = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImages/img"
# directory_style = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImages/style/"



# def checkWhetherDimensionAreDifferent(directory):
#     images = os.listdir(directory)
#     for image in images:
#         img = Image.open(os.path.join(directory,image))
#         if img.height != 512 or img.width !=512:
#             print(image)

    




# checkWhetherDimensionAreDifferent(directory=directory_colored)
# checkWhetherDimensionAreDifferent(directory=directory_style)
# checkWhetherDimensionAreDifferent(directory=directory_black_and_white)


# import os
# import random
# import shutil

# # Set random seed for reproducibility (optional)
# random.seed(42)

# # Define source (training) directories
# train_content_dir = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImages/img"
# train_style_dir = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImages/style"
# train_mask_dir = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImages/mask"

# # Define destination (test) directories
# test_content_dir = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImagesTest/img"
# test_style_dir = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImagesTest/style"
# test_mask_dir = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImagesTest/mask"

# # Create destination directories if they do not exist
# os.makedirs(test_content_dir, exist_ok=True)
# os.makedirs(test_style_dir, exist_ok=True)
# os.makedirs(test_mask_dir, exist_ok=True)

# # List all image files in the training content folder.
# # Assuming your images are .png files. Adjust the extension if needed.
# all_files = [f for f in os.listdir(train_content_dir) if f.endswith(".png")]

# # Calculate 20% of the files
# sample_size = int(0.2 * len(all_files))
# print(f"Total images: {len(all_files)}; moving {sample_size} images to the test folder.")

# # Randomly select 20% of the files
# selected_files = random.sample(all_files, sample_size)

# # Move the selected files from each folder to the corresponding test folder.
# for filename in selected_files:
#     # Construct full paths for the content images
#     src_content = os.path.join(train_content_dir, filename)
#     dst_content = os.path.join(test_content_dir, filename)
#     if os.path.exists(src_content):
#         shutil.move(src_content, dst_content)
#     else:
#         print(f"File not found: {src_content}")
        
#     # Construct full paths for the style images
#     src_style = os.path.join(train_style_dir, filename)
#     dst_style = os.path.join(test_style_dir, filename)
#     if os.path.exists(src_style):
#         shutil.move(src_style, dst_style)
#     else:
#         print(f"File not found: {src_style}")
        
#     # Construct full paths for the mask (colored) images
#     src_mask = os.path.join(train_mask_dir, filename)
#     dst_mask = os.path.join(test_mask_dir, filename)
#     if os.path.exists(src_mask):
#         shutil.move(src_mask, dst_mask)
#     else:
#         print(f"File not found: {src_mask}")


# import torchvision.transforms as transforms
# from PIL import Image

# # Load an image
# img = Image.open('image.jpg')

# # Apply RandomCrop transform
# cropped_img = transforms.RandomCrop(224)(img)

# # Both the original image and the cropped image exist
# print(img.size)  # prints the original image size
# print(cropped_img.size)  # prints the size of the cropped image (224x224)


from PIL import Image
import os

# Directory containing the images
image_dir = 'preprocessedImages/img'

# List of image files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Check the first 5 images
for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    with Image.open(img_path) as img:
        if(img.mode == "L"):
            print("here")

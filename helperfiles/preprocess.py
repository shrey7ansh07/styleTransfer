# from PIL import Image
# import os
# def get_image_dimensions(image_path):
#     with Image.open(image_path) as img:
#         return img.size

# # Example usage


# directory = "dataset/img"
# imageList = os.listdir(directory)
    


# def parseImagesWithDifferentDimension(output_path):
#     for imagePath in imageList:
#         actualPath = directory+"/"+imagePath
#         width, height = get_image_dimensions(image_path=actualPath)
#         img = Image.open(actualPath)
#         newHeight = max(512, height)
#         newWidth = max(512,width)
#         new_img = Image.new("RGB", (newWidth, newHeight), (255, 255, 255))
#         top_left_x = (newWidth - width) // 2
#         top_left_y = (newHeight - height) // 2
#         new_img.paste(img, (top_left_x, top_left_y))
#         new_img.save(output_path+imagePath)



# output_path = "preprocessedImages/img/"
# parseImagesWithDifferentDimension(output_path=output_path) 


from PIL import Image
import numpy as np

def is_grey_scale(img_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    return img

print(is_grey_scale("/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/testImage/blackWhite.png"))

from PIL import Image
import os
import random
import numpy as np

directory_colored = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/dataset/colored"
directory_black_and_white = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImages/img"
saveFolderDirectory = "/Users/shreyansh/Desktop/style transfer/ADAIN:architecture/preprocessedImages/style/"
colored_images = os.listdir(directory_colored)
blackAndWhiteImage = os.listdir(directory_black_and_white)
cnt = 0
for images in os.listdir(directory_black_and_white):
    selectedImageIndex = random.choice(colored_images)
    selectedImagePath = os.path.join(directory_colored, selectedImageIndex)
    styleImage = Image.open(selectedImagePath)
    # print(images)
    styleImage.save(os.path.join(saveFolderDirectory, images))
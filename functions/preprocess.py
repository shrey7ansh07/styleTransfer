from PIL import Image

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size

# Example usage
image_path = 'testImage/blackWhite.jpeg'
width, height = get_image_dimensions(image_path)
print(f"Width: {width}, Height: {height}")



def parseImagesWithDifferentDimension(imagePath):
    width, height = get_image_dimensions(image_path=imagePath)
    

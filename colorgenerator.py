from PIL import Image
import random

def generate_random_grid():
    # Create a new image with RGB mode
    img = Image.new('RGB', (512, 512))
    
    # Calculate slot size
    slot_size = 512 // 8  # 64 pixels per slot
    
    # Get pixel access object
    pixels = img.load()  # This is a list of pixels
                
    # Generate random colors for each slot
    for row in range(8):
        for col in range(8):
            # Generate random RGB color
            color = (
                random.randint(0, 255),  # Red
                random.randint(0, 255),  # Green
                random.randint(0, 255)   # Blue
            )
            
            # Fill the entire slot with the random color
            for x in range(col * slot_size, (col + 1) * slot_size):
                for y in range(row * slot_size, (row + 1) * slot_size):
                    pixels[x, y] = color
    
    return img

# Generate and save the image
if __name__ == "__main__":
    image = generate_random_grid()
    image.save("random_grid.png")
    print("Image generated successfully as 'random_grid.png'")
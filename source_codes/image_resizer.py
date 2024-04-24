# Import necessary libraries
import os
import cv2


# Define a class to resize images
class ImageResizer:
    # Initialize with directory path
    def __init__(self, dir_path):
        self.dir_path = dir_path

    # Method to resize images
    def resize_images(self):
        # Iterate over all files in the directory
        for filename in os.listdir(self.dir_path):
            # Check if the file is a .JPG image
            if filename.endswith(".JPG"):
                # Read the image
                image = cv2.imread(os.path.join(self.dir_path, filename))
                # Resize the image
                resized = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                # Save the resized image
                cv2.imwrite(os.path.join(self.dir_path, filename), resized)


# Main function
if __name__ == "__main__":
    # Create an instance of ImageResizer for the current working directory
    resizer = ImageResizer(os.getcwd())
    # Call the resize_images method on the instance
    resizer.resize_images()

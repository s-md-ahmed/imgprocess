import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from skimage import io
# Load the image file
image_path = "C:\Python\cat.jpg"  # Replace with the path to your image
image_data = io.imread(image_path)
# Remove the alpha channel if present (if image has 4 channels - RGBA, we keep only RGB)
if image_data.shape[2] == 4:
    image_data = image_data[:, :, :3]
# Normalize the image data to the range [0, 1] for each color channel to simplify PCA
normalized_image_data = image_data / 255.0#max size of each pixel is 255 thats why we divide by 255 for the width height and colour values max values is 255
max_components = min(normalized_image_data.shape[0], normalized_image_data.shape[1])
print("Maximum number of components =", max_components)
# Perform PCA separately on each color channel
reconstructed_color_image = np.zeros_like(normalized_image_data)#create an array of same dimensions as normalised image data with 0's
# Loop over each color channel (Red, Green, and Blue)
for channel in range(3):#run a for loop to work on all the the rgb colour channels
    pca = PCA(n_components=110)  # Create a PCA object with 200 components
    pca.fit(normalized_image_data[:, :, channel])  # Fit PCA to the current color
    #channel to calculate the principal component values
    # Transform the image data using PCA to reduce dimensionality
    new_image_data = pca.transform(normalized_image_data[:, :, channel])
    # Reconstruct the image data from the reduced data using inverse transform
    reconstructed_channel = pca.inverse_transform(new_image_data)
    # Ensure the reconstructed channel is in the valid range [0, 1]
    reconstructed_channel = np.clip(reconstructed_channel, 0, 1)
    # Assign the reconstructed channel back to the appropriate color channel in the final image
    reconstructed_color_image[:, :, channel] = reconstructed_channel
# Visualize the original image and the reconstructed image side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_data)
axes[0].set_title("Original Image")
axes[0].axis('off')
axes[1].imshow(reconstructed_color_image)
axes[1].set_title("Reconstructed Image")
axes[1].axis('off')
plt.show()
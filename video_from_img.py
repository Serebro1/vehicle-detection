import os
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Set path to the folder with images
path = "data/"

# Counting the number of images in the directory
image_files = [file for file in os.listdir(path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
num_of_images = len(image_files)
if num_of_images == 0:
    raise ValueError("No images found in the specified directory.")

# Calculating the mean width and height of all images
mean_width = 0
mean_height = 0

for file in image_files:
    file_path = os.path.join(path, file)
    with Image.open(file_path) as im:
        width, height = im.size
        mean_width += width
        mean_height += height

# Averaging width and height
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)


def resize_image(file):
    """Resizes an image to the mean width and height."""
    file_path = os.path.join(path, file)
    output_path = os.path.join("data_frames/", f"resized_{file}")
    try:
        with Image.open(file_path) as im:
            im_resized = im.resize((mean_width, mean_height), Image.LANCZOS)
            im_resized.save(output_path, 'JPEG', quality=95)
    except Exception as e:
        # Logging can be added here if needed
        pass


# Resizing images in parallel with progress bar
# print("Resizing images...")
# with ThreadPoolExecutor() as executor:
#     list(tqdm(executor.map(resize_image, image_files), total=num_of_images, desc="Resizing"))


# Function to generate video
def generate_video():
    video_name = os.path.join("videos", 'mygeneratedvideo.mp4')
    images = [img for img in os.listdir("data/") if img.endswith(".jpg")]

    if not images:
        raise ValueError("No resized images found for video generation.")

    # Sort images to ensure the correct order
    #images.sort()

    # Set frame from the first image
    first_image_path = os.path.join("data/", images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (width, height))

    # Appending images to video with progress bar
    for image in tqdm(images, desc="Creating video"):
        image_path = os.path.join("data/", image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video file
    video.release()
    
    print("Video generated successfully at:", video_name)


# Calling the function to generate the video
generate_video()
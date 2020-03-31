import cv2
import os

image_folder = "images/acai/x_hat"
video_name = "videos/x_hat.avi"

images = list(sorted(os.listdir(image_folder)))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width, height))

# AH. Bigger numbers are higher.
# Unless we 0-padded.

for image in images:

    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
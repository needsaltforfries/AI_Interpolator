import os
import numpy as np
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

true_frames_path = "./clip2/"
fake_frames_path = "./video/gen_frames2/"

directories = []
for i in range(len(os.listdir(true_frames_path)) - 2):
    img = os.listdir(true_frames_path)[i]
    img2 = os.listdir(fake_frames_path)[i]
    if img.lower().endswith(('jpeg')):
        truePath = os.path.join(true_frames_path, img)
        fakePath = os.path.join(fake_frames_path, img2)

        directories.append(truePath)
        directories.append(fakePath)
        
img = os.listdir(true_frames_path)[len(os.listdir(true_frames_path)) - 1]
print("loaded images")

outpath = './video/clip2.mp4'
rate = 120
frame_size = (1280, 720)
writer = FFMPEG_VideoWriter(outpath, frame_size, fps=rate, codec="libx264")

for i in range(len(directories)):
    frame = Image.open(directories[i])
    frame = frame.resize((1280, 720))
    print(f"\rworking on frame {i}/{len(directories)}", end= "")
    writer.write_frame(frame)
    frame.close()
print("\nSuccess!")
writer.close()
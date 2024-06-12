from moviepy.editor import ImageSequenceClip

original_frames = []
stabilized_images = []
cropped_images = []

for i in range(0, 101):
    num = str(i)
    num = ((3- len(num)) * '0') + num
    original_frames.append(f"input/Frame{num}.jpg")
    stabilized_images.append(f"build/COMPLETION/stable{num}.png")
    cropped_images.append(f"build/CHALLENGE/crop{num}.png")

print("Original")
clip = ImageSequenceClip(original_frames, fps=10)
clip.write_videofile("results/original.mp4", codec="libx264", fps=20)

print("stable")
clip = ImageSequenceClip(stabilized_images, fps=10)
clip.write_videofile("results/stabilized.mp4", codec="libx264", fps=20)

print("crop")
clip = ImageSequenceClip(cropped_images, fps=10)
clip.write_videofile("results/cropped.mp4", codec="libx264", fps=20)



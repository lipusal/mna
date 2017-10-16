import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description="Heart rate monitor. Uses Fourier Transform to estimate beats per minute"
                                             "based off of a video.")
parser.add_argument("video", help="Video to analyze", type=str)
parser.add_argument("--size", "-s", help="Size of observation window, in pixels. Window is always centered. Default "
                                         "is 30",
                    type=int, default=30)
parser.add_argument("--verbose", "-v", help="Print verbose information while running", action="store_true",
                    default=False)
parser.add_argument("--time", "-t", help="Print elapsed program time", action="store_true", default=False)
args = parser.parse_args()

if args.time:
    import mna.util.Timer

# noinspection PyArgumentList
video = cv2.VideoCapture(args.video)
if not video.isOpened():
    print("Couldn't open video. Aborting.")
    exit(1)

SIZE = args.size
WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
NUM_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = video.get(cv2.CAP_PROP_FPS)

if not 0 < args.size < min(WIDTH, HEIGHT):
    print("Invalid size. Allowed values for this video are between 0 and %i, both ends exclusive." % min(WIDTH, HEIGHT))
    exit(1)

if args.verbose:
    print("Read %i frames of %ix%i each; FPS = %g" % (NUM_FRAMES, WIDTH, HEIGHT, FPS))

# R,G,B means for each frame
r = np.zeros((1, NUM_FRAMES))
g = np.zeros((1, NUM_FRAMES))
b = np.zeros((1, NUM_FRAMES))

# Capture a square of the specified size, centered
left_bound = WIDTH//2 - SIZE//2
right_bound = WIDTH//2 + SIZE//2
upper_bound = HEIGHT//2 + SIZE//2
lower_bound = HEIGHT//2 - SIZE//2
k = 0
while video.isOpened():
    did_read, frame = video.read()

    if did_read:
        r[0, k] = np.mean(frame[left_bound:right_bound, lower_bound:upper_bound, 0])
        g[0, k] = np.mean(frame[left_bound:right_bound, lower_bound:upper_bound, 1])
        b[0, k] = np.mean(frame[left_bound:right_bound, lower_bound:upper_bound, 2])
    # print(k)
    else:
        break
    k += 1

video.release()             # Close the video
cv2.destroyAllWindows()     # Close any windows opened behind the scenes

n = 1024
f = np.linspace(-n / 2, n / 2 - 1, n) * FPS / n

# Subtract mean to each color (the first index is necessary because these have shape (1, 1843) not (,1843) -- yes, there is no number before the second comma
r = r[0, 0:n] - np.mean(r[0, 0:n])
g = g[0, 0:n] - np.mean(g[0, 0:n])
b = b[0, 0:n] - np.mean(b[0, 0:n])

# Apply Fast Fourier Transform
R = np.abs(np.fft.fftshift(np.fft.fft(r))) ** 2
G = np.abs(np.fft.fftshift(np.fft.fft(g))) ** 2
B = np.abs(np.fft.fftshift(np.fft.fft(b))) ** 2

plt.plot(60 * f, R)
plt.xlim(0, 200)

plt.plot(60 * f, G)
plt.xlim(0, 200)
plt.xlabel("frecuencia [1/minuto]")

plt.plot(60 * f, B)
plt.xlim(0, 200)

print("Frecuencia cardíaca: ", abs(f[np.argmax(G)]) * 60, " pulsaciones por minuto")

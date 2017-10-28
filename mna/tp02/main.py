import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mna.tp02.fft import original, base_recursive

parser = argparse.ArgumentParser(description="Heart rate monitor. Uses Fourier Transform to estimate beats per minute"
                                             "based off of a video.")
parser.add_argument("video", help="Video to analyze", type=str)
parser.add_argument("--size", "-s", help="Size of observation window, in pixels. Window is always centered. Default "
                                         "is 30",
                    type=int, default=30)
parser.add_argument("--start", help="Video time at which to start capturing frames. Decimal.", type=float, default=0)
parser.add_argument("--end", help="Video time at which to stop capturing frames. Decimal.", type=float, default=-1)
parser.add_argument("--verbose", "-v", help="Print verbose information while running", action="store_true",
                    default=False)
parser.add_argument("--time", "-t", help="Print elapsed program time", action="store_true", default=False)
args = parser.parse_args()

if args.time:
    import mna.util.timer

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
    print("Opened video with %i frames (%fs) of %ix%i each; FPS = %g" % (NUM_FRAMES, NUM_FRAMES/FPS, WIDTH, HEIGHT, FPS))

# R,G,B means for each frame
r = np.zeros((1, NUM_FRAMES))
g = np.zeros((1, NUM_FRAMES))
b = np.zeros((1, NUM_FRAMES))

# Capture a square of the specified size, centered
left_bound = WIDTH//2 - SIZE//2
right_bound = WIDTH//2 + SIZE//2
upper_bound = HEIGHT//2 + SIZE//2
lower_bound = HEIGHT//2 - SIZE//2
current_frame = 0
current_time = 0
start_time = args.start
end_time = args.end if args.end != -1 else NUM_FRAMES/FPS

if args.verbose:
    print("Reading %i frames..." % ((end_time - start_time) * FPS), end='', flush=True)

while video.isOpened():
    did_read, frame = video.read()

    if did_read:
        if start_time <= current_time <= end_time:
            b[0, current_frame] = np.mean(frame[left_bound:right_bound, lower_bound:upper_bound, 0])
            g[0, current_frame] = np.mean(frame[left_bound:right_bound, lower_bound:upper_bound, 1])
            r[0, current_frame] = np.mean(frame[left_bound:right_bound, lower_bound:upper_bound, 2])
        elif current_time > end_time:
            break
    # print(k)
    else:
        break
    current_frame += 1
    current_time += 1/FPS

video.release()             # Close the video
cv2.destroyAllWindows()     # Close any windows opened behind the scenes
if args.verbose:
    print("done")
    print("Normalizing data...")

# Plot RGB across simulation
x = np.arange(NUM_FRAMES)
plt.plot(x, r[0], color='red')
plt.plot(x, g[0], color='green')
plt.plot(x, b[0], color='blue')

# Discretize simulation interval in N parts, where N is the biggest power of 2 that fits in NUM_FRAMES
n = int(2 ** np.floor(np.log2(NUM_FRAMES)))
f = np.linspace(-n / 2, n / 2 - 1, n) * FPS / n

# Subtract mean to each color (the first index is necessary because these have shape (1, 1843) not (,1843) -- yes, there is no number before the second comma
r = r[0, 0:n] - np.mean(r[0, 0:n])
g = g[0, 0:n] - np.mean(g[0, 0:n])
b = b[0, 0:n] - np.mean(b[0, 0:n])

# Apply Fast Fourier Transform with the desired algorithm
# TODO: Pass variant as parameter?
fft = base_recursive.fft
fftshift = base_recursive.fftshift

if args.verbose:
    print("Applying FFT...", end='')

R = np.abs(fftshift(fft(r))) ** 2
G = np.abs(fftshift(fft(g))) ** 2
B = np.abs(fftshift(fft(b))) ** 2
# Originals for comparing TODO remove when done
r2 = np.abs(original.fftshift(original.fft(r))) ** 2
g2 = np.abs(original.fftshift(original.fft(g))) ** 2
b2 = np.abs(original.fftshift(original.fft(b))) ** 2

if args.verbose:
    print("done")
    print("Max error against original FFT: %g" % np.max(np.abs([R-r2, G-g2, B-b2])))

plt.plot(60 * f, R, color='red')
plt.xlim(0, 200)

plt.plot(60 * f, G, color='green')
plt.xlim(0, 200)
plt.xlabel("frecuencia [1/minuto]")

plt.plot(60 * f, B, color='blue')
plt.xlim(0, 200)

# Normal human heartbeat is within 40, 120 BPM. Only study frequencies within that window
# TODO: Shorten
R_trim, G_trim, B_trim = [], [], []
for i in range(len(f)):
    freq = f[i] * 60
    if 40 <= freq <= 120:
        R_trim.append(R[i])
        G_trim.append(G[i])
        B_trim.append(B[i])
    else:
        R_trim.append(0)
        G_trim.append(0)
        B_trim.append(0)

print("Frecuencia cardíaca (R): ", abs(f[np.argmax(R_trim)]) * 60, " pulsaciones por minuto")
print("Frecuencia cardíaca (G): ", abs(f[np.argmax(G_trim)]) * 60, " pulsaciones por minuto")
print("Frecuencia cardíaca: (B)", abs(f[np.argmax(B_trim)]) * 60, " pulsaciones por minuto")

print("Done")

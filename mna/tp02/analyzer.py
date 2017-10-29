import argparse
import numpy as np
import matplotlib
# Use a non-interactive backend for matplotlib so figures aren't shown, only exported (must be done before importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from mna.tp02.fft import original, base_recursive
from mna.tp02.filter import heartrate
import os
from os import path
import math

parser = argparse.ArgumentParser(description="Heart rate monitor. Uses Fourier Transform to estimate beats per minute"
                                             "based off of a video.")
parser.add_argument("video", help="Video to analyze", type=str)
parser.add_argument("--output", "-o", help="Base output directory in which to save output", type=str, default=".")
parser.add_argument("--size", "-s", help="Size of observation window, in pixels. Window is always centered. Default "
                                         "is 30",
                    type=int, default=30)
parser.add_argument("--start", help="Video time at which to start capturing frames. Decimal.", type=float, default=0)
parser.add_argument("--end", help="Video time at which to stop capturing frames. Decimal.", type=float, default=-1)
args = parser.parse_args()

# Create output dir if necessary
if not path.exists(args.output):
    os.makedirs(args.output)

# Create CSV file headers
csv = open(path.normpath(path.join(args.output, "heart_rates.csv")), "w")
csv.write("Duration,Window Size,R,G,B\n")

print("******************************************************************")
print("Analyzing %s" % args.video)
print("******************************************************************")

# Argument combinations
# DURATIONS = [1, 5, 10, 20, 30, 60, -1]    # -1 = to end of video
DURATIONS = [1, 5, 10, 20, 30, 60]
SIZES = [15, 30, 50, 100, 200, 500]

for duration in DURATIONS:
    for size in SIZES:
        print("------------------------------------------------------------------")
        print("Run with duration = %i, window size = %i" % (duration, size))
        print("------------------------------------------------------------------")

        run_prefix = path.normpath(path.join(args.output, "d%02i_s%02i" % (duration, size)))

        # noinspection PyArgumentList
        video = cv2.VideoCapture(args.video)
        if not video.isOpened():
            print("Couldn't open video. Aborting.")
            exit(1)

        # Override size
        args.size = size

        SIZE = args.size
        WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        NUM_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        FPS = video.get(cv2.CAP_PROP_FPS)

        # Override duration
        args.end = duration

        if not 0 < args.size < min(WIDTH, HEIGHT):
            print("Invalid size. Allowed values for this video are between 0 and %i, both ends exclusive." % min(WIDTH, HEIGHT))
            exit(1)

        # Capture a square of the specified size, centered
        left_bound = WIDTH//2 - SIZE//2
        right_bound = WIDTH//2 + SIZE//2
        upper_bound = HEIGHT//2 + SIZE//2
        lower_bound = HEIGHT//2 - SIZE//2
        current_frame = 0
        current_time = 0
        # Calculate start/end times and frames
        start_time = args.start
        end_time = args.end if args.end != -1 else NUM_FRAMES/FPS
        end_frame = int(math.ceil(end_time * FPS))

        # R,G,B means for each frame
        r = np.zeros((1, end_frame))
        g = np.zeros((1, end_frame))
        b = np.zeros((1, end_frame))

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

        # Plot RGB across video
        plt.figure("R, G, B en el Video")
        plt.suptitle("R, G, B en el Video")
        plt.xlabel("# Frame")
        plt.ylabel("Valor")

        x = np.arange(start_time*FPS, end_time*FPS)
        plt.plot(x, r[0], color='red')
        plt.plot(x, g[0], color='green')
        plt.plot(x, b[0], color='blue')
        # Export
        plt.savefig("%s_rgb.png" % run_prefix, bbox_inches='tight')
        plt.close()

        # Discretize simulation interval in N parts, where N is the biggest power of 2 that fits in NUM_FRAMES
        n = int(2 ** np.floor(np.log2(end_frame)))
        f = np.linspace(-n / 2, n / 2 - 1, n) * FPS / n

        # Subtract mean to each color (the first index is necessary because these have shape (1, 1843) not (,1843) -- yes, there is no number before the second comma
        r = r[0, 0:n] - np.mean(r[0, 0:n])
        g = g[0, 0:n] - np.mean(g[0, 0:n])
        b = b[0, 0:n] - np.mean(b[0, 0:n])

        # Apply Fast Fourier Transform with the desired algorithm
        # TODO: Pass variant as parameter?
        fft = base_recursive.fft
        fftshift = base_recursive.fftshift

        R = np.abs(fftshift(fft(r))) ** 2
        G = np.abs(fftshift(fft(g))) ** 2
        B = np.abs(fftshift(fft(b))) ** 2
        # Originals for comparing TODO remove when done
        r2 = np.abs(original.fftshift(original.fft(r))) ** 2
        g2 = np.abs(original.fftshift(original.fft(g))) ** 2
        b2 = np.abs(original.fftshift(original.fft(b))) ** 2

        # Plot FFT result
        plt.figure("FFT para cada Color")
        plt.suptitle("FFT para cada Color")
        plt.xlabel("Frecuencia [pulsaciones/minuto]")
        plt.ylabel("60 * abs(fftshift(fft(color))) ^ 2")

        # ######################################################################################################################
        # Plot the entire observed frequency range, but analyze a filtered version of it
        # ######################################################################################################################
        # Plot limits
        PLOT_LOWER, PLOT_UPPER = 0, 200
        # Analyzed limits (normal human heartbeat range)
        LOWER, UPPER = 40, 120

        plt.plot(60 * f, R, color='red')
        plt.xlim(PLOT_LOWER, PLOT_UPPER)
        plt.plot(60 * f, G, color='green')
        plt.xlim(PLOT_LOWER, PLOT_UPPER)
        plt.plot(60 * f, B, color='blue')
        plt.xlim(PLOT_LOWER, PLOT_UPPER)
        plt.xlabel("Frecuencia [pulsaciones/minuto]")
        # Vertical lines for analyzed range
        plt.axvline(x=LOWER, linestyle="--")
        plt.axvline(x=UPPER, linestyle="--")

        # Filter frequencies
        R_filtered = heartrate.filter(R, f * 60, UPPER, LOWER)
        G_filtered = heartrate.filter(G, f * 60, UPPER, LOWER)
        B_filtered = heartrate.filter(B, f * 60, UPPER, LOWER)


        # FFT maximum = most similarity, in frequency, to a sinusoid. FFT maximum * 60 = most similarity in beats per minute
        fft_max_r = np.argmax(R_filtered)   # Frequency at which maximum occurs
        fft_max_g = np.argmax(G_filtered)
        fft_max_b = np.argmax(B_filtered)
        heartrate_r = abs(f[fft_max_r]) * 60
        heartrate_g = abs(f[fft_max_g]) * 60
        heartrate_b = abs(f[fft_max_b]) * 60
        print("Frecuencia cardíaca (R): ", heartrate_r, " pulsaciones por minuto")
        print("Frecuencia cardíaca (G): ", heartrate_g, " pulsaciones por minuto")
        print("Frecuencia cardíaca (B): ", heartrate_b, " pulsaciones por minuto")
        # Plot maxima as points
        plt.plot(heartrate_r, R[fft_max_r], marker="o", color="red")
        plt.plot(heartrate_g, G[fft_max_g], marker="o", color="green")
        plt.plot(heartrate_b, B[fft_max_b], marker="o", color="blue")
        # Export
        plt.savefig("%s_fft.png" % run_prefix, bbox_inches='tight')
        plt.close()

        # Export estimated heartrates
        csv.write("%i,%i,%f,%f,%f\n" % (duration, size, heartrate_r, heartrate_g, heartrate_b))
        csv.flush()

csv.close()
print("******************************************************************")
print("Done")
print("******************************************************************")

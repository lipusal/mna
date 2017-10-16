"""
Original, numpy-based implementation of Fast Fourier Transform
"""
import numpy as np


def fft(data):
    return np.abs(np.fft.fftshift(np.fft.fft(data))) ** 2

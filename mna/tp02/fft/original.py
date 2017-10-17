"""
Original, numpy-based implementation of Fast Fourier Transform
"""
import numpy as np


def fft(data):
    return np.fft.fft(data)


def fftshift(data):
    return np.fft.fftshift(data)

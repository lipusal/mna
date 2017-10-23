"""
Base Cooley-Tukey recursive implementation, inspired from in-class notes and pseudocode found online, eg.
https://rosettacode.org/wiki/Fast_Fourier_transform#Python
https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Pseudocode
"""
from math import pi
from cmath import exp   # To be able to use complex numbers in exp()


def fft(data):
    n = len(data)
    if n == 1:
        return data
    elif n % 2 != 0:
        raise Exception("Need an even-length data vector for Cooley-Tukey FFT, provided data has length %i" % n)

    # Divide - do FFT of even elements and odd elements
    even_half = fft(data[0::2])
    odd_half = fft(data[1::2])
    # Conquer (NOTE: i, ie. sqrt(-1), is expressed as j in Python, it's electrical engineering notation)
    odd_terms = [exp(-2j * pi * k / n) * odd_half[k] for k in range(n // 2)]
    return \
        [even_half[k] + odd_terms[k] for k in range(n // 2)] + \
        [even_half[k] - odd_terms[k] for k in range(n // 2)]


def fftshift(data):
    n = len(data)
    return data[n//2:n] + data[0:n//2]

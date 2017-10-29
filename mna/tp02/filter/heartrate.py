def filter(fft, frequencies, upper_limit, lower_limit):
    """Filter out FFT coefficients corresponding to frequencies outside the specified range (ie. set them to 0)"""
    result = []
    for i in range(len(frequencies)):
        if lower_limit <= frequencies[i] <= upper_limit:
            result.append(fft[i])
        else:
            result.append(0)

    return result

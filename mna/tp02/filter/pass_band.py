def low_pass_filter(signal, upper_limit):
    """Pass signals lower than the upper limit, filter out those higher than the upper limit."""
    return [datum if datum <= upper_limit else 0 for datum in signal]


def high_pass_filter(signal, lower_limit):
    """Pass signals higher than the lower limit, filter out those lower than the lower limit."""
    return [datum if datum >= lower_limit else 0 for datum in signal]

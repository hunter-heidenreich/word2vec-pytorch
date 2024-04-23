def num2str(num):
    """Represent a number as a string.

    Specifically, round to the nearest grouping:
    - K for thousands
    - M for millions
    - B for billions
    - T for trillions

    If the number is less than 1000, return the number as a string.

    Args:
        num: The number to represent.

    Returns:
        The number as a string.
    """
    if num < 1000:
        return str(num)

    num = float(num)
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f"{num:.1f}{'KMBT'[magnitude - 1]}"

def add_ints(a, b):
    return int(a) + int(b)


def format_person(name, age, height):
    return f"{name} is {age} years old, {height:.2f}m tall"


def weighted_sum(values, weights):
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length")
    total = 0.0
    for value, weight in zip(values, weights):
        total += float(value) * float(weight)
    return total

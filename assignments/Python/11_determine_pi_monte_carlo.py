from typing import List

from math import pi
from random import random
from sys import argv
from typing import List

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return pi * self.radius**2

    def is_point_inside_circle(self, x, y):
        z = x**2 + y**2
        return z <= self.radius**2

    def generate_random_coordinate(self):
        x = random()
        y = random()
        return x, y

def usage():
    print("Usage: python3 09_determine_pi_monte_carlo.py <number of tries>")
    return 1

def main(argv: List[str]):
    num_tries = int(argv[0])

    # Create an instance of the Circle class representing the unit circle
    unit_circle = Circle(1.0)

    # Generate random points and count how many fall inside the unit circle
    in_circle = 0
    for _ in range(num_tries):
        x, y = unit_circle.generate_random_coordinate()

        if unit_circle.is_point_inside_circle(x, y):
            in_circle += 1

    # Estimate pi based on the ratio of points inside the unit circle
    estimated_pi = in_circle * 4.0 / num_tries

    # Print the estimated value of pi
    print(f"Estimated value of pi: {estimated_pi}")

if __name__ == "__main__":
    if(len(argv)) != 2:
        usage()

    main(argv[1:])

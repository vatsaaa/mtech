'''Assignment Question: Given an array which may contain duplicates, print all elements and their frequencies'''
from pprint import pprint
user_input = list(map(int, input("Please enter a list of numbers separated by space: ").split()))

count_map = {}

for n in user_input:
    if n in count_map:
        count_map[n] = count_map[n] + 1
    else:
        count_map[n] = 1

pprint(count_map)

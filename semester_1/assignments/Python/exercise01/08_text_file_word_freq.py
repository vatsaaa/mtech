'''
Assignment Question: Parse a user given file and display
the occurrence frequency of each word in the text within 
the file. Also, disolay the 3 top most frequent words.
'''
import re

# Open the file and get a file-handle
file = open("./inputs/08.txt", "r")

words = {}

for line in file:
    # Remove the punctuation marks and split the line into words
    new_line = re.sub(r'[^\w\s]','', line.strip()).lower().split(" ")

    for word in new_line:
        if word in words:
            words[word] = words[word] + 1
        else:
            words[word] = 1

# Now sort the dictionary on values instead of keys
sorted_words = sorted(words.items(), key=lambda x:x[1])

# Print the top 3
for i in range(-1,-4, -1):
    print(f"Word {sorted_words[i][0]} appears for {sorted_words[i][1]} times")

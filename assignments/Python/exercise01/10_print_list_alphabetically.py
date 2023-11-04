# Prompt user to enter a list of names
names = input("Enter a list of names separated by commas: ")

# Split the input string into a list of names
names_list = names.split(",")

# Strip any leading or trailing whitespaces from each name in the list
names_list = [name.strip() for name in names_list]

# Sort the list of names in alphabetical order
names_list.sort()

# Display the sorted list of names
print("Sorted list of names:")
for name in names_list:
    print(name)

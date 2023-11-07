'''Assignmemt Question: Palindrome'''
input_as_array = [*input("Please enter a string: ")]

input_len = len(input_as_array)

for i in range(0, input_len, 1):
    if(input_as_array[i] == input_as_array[input_len - i - 1]):
        continue
    else:
        print(input_as_array, "is not a palindrome!")
        exit(-1)

print(''.join(input_as_array), "is a palindrome!!")


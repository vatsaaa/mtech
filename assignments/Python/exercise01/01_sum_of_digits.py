'''Assignment Question: For a user give number, print the sum of its digits'''
num = input("Please enter a multi-digit number:")

digits = [*num]
digit_sum = 0

for d in digits:
	digit_sum = digit_sum + int(d)

print("Sum of digits of number ", num, " is: ", digit_sum)

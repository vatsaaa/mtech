'''Assignment Question: Armstrong number'''
num = input("Please enter a multi-digit number: ")
digits = [*num]

order = int(input("Please enter order: "))

digit_sum_to_order = 0

for d in digits:
    digit_sum_to_order = digit_sum_to_order + int(d)**order

if digit_sum_to_order == int(num):
    print(num, " is an Armstrong number of order", order)
else:
    print(num, " is not an Armstrong number!")
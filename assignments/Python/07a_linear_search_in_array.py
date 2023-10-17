'''Assignment Question: Given arrray of n elements, search for element X in it'''

str_arr = input("Please give the array to seach within: ")
arr = str_arr.split(' ')

element = int(input("Give the element to search in this array: "))

count = 0

for a in arr:
    if(int(a) == element):
        print("Found", element, f"at arr[{count}]")
        break
    else:
        count = count + 1
        continue

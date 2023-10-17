'''Assignment Question: Given a number n find all of its prime factors'''
from math import floor, sqrt

def is_prime(numb: int):
    for d in range(2, floor(sqrt(numb))+1, 1):
        if(numb % d == 0):
            return False
        else:
            continue
    
    return True

def find_primes_upto(numb: int):
    primes = []
    for d in range(2, numb, 1):
        if(is_prime(d)):
            primes.append(d)

    return primes

def main():
    input_num = int(input("Please enter the number whose prime factors are required: "))

    primes_to = find_primes_upto(input_num)

    prime_factors = []

    for p in primes_to:
        if(input_num % p == 0):
            prime_factors.append(p)
    
    print(prime_factors)

if __name__ == '__main__':
    main()

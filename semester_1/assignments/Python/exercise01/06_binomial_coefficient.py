def factorial(n: int):
    fact = 1

    for i in range(1, n+1, 1):
        fact = fact * i

    return fact

def main(n: int, r: int):
    n_factorial = factorial(n)
    r_factorial = factorial(r)
    n_minus_r_factorial = factorial(n - r)

    binomial_coefficient = (n_factorial // (r_factorial * n_minus_r_factorial))

    print(f"C({n}, {r}) is", binomial_coefficient)

if __name__ == '__main__':
    n = int(input("Please input n: "))
    r = int(input("Please input r: "))

    if (n < r):
        print(f"C({n}, {r}) is not defined!")

    main(n, r)
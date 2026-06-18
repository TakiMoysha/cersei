import sys

def gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b != 0:
        a, b = b, a % b
    return a

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)
    
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    print(gcd(a, b))

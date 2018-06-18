import numpy as np

A = [
     [1, 2],
     [3, 4],
     [9, 6]
     ]

B = np.matrix([
               [2, 3, 5],
               [7, 11, 13],
               ])



y = np.ones((2, 3))
print(y)

print(y.shape[0])


def uno(x):
    return 1*x

def due(x):
    return 2*x

def tre(x):
    return 3*x

def somma(funzioni, x):
    sum = 0
    for funzione in funzioni:
        sum += funzione(x)
    print(sum)



        

from random import random
import time
import matplotlib.pyplot as plt 
import seaborn as sns

def random_points():
    x = 2*random() - 1
    y = 2*random() - 1
    return (x, y)

def PiCalculator(n : int, want_a_plot=False):
    
    print("Calculating pi...")

    points_list = []

    for i in range(n):
        a, b = random_points()
        points_list.append((a, b))

    X = []
    Y = []

    for i in range(len(points_list)):
        X.append(points_list[i][0])
        Y.append(points_list[i][1])

    X_circle = [x for x,y in zip(X,Y) if x**2 + y**2 <= 1]
    Y_circle = [y for x,y in zip(X,Y) if x**2 + y**2 <= 1]

    circle_points = [(X_circle[i], Y_circle[i]) for i in range(len(X_circle))]

    print(f"The value of pi is approximately equal to: {(len(X_circle)/len(X))*4}")

    if (want_a_plot==True):
        plt.figure(figsize=(5,5))
        sns.scatterplot(x=X, y=Y, size=1.5)
        plt.plot(X_circle, Y_circle, 'ro', markersize=1.2)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()

if __name__ == "__main__":
    start = time.time()
    PiCalculator(n=1000000)
    end = time.time()
    print(f"The calculation took about {end-start} seconds")
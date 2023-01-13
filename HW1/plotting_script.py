
import matplotlib.pyplot as plt
import sys

fig_name = sys.argv[1] 
X = [90, 50, 25, 10, 5]
f = open("fptree.txt")
Y1 = list(map(float, f.readline().strip().split())) #FP-Tree Algorithm Times
f.close()

a = open("apriori.txt")
Y2 = list(map(float, a.readline().strip().split())) #Apriori Algorithm Times
a.close()

plt.plot(X, Y1, color='r', label = "FP-Tree Algorithm")
plt.plot(X, Y2, color='g', label = "Apriori Algorithm")

plt.xlabel("Support Threshold")
plt.ylabel("Running Times in milli seconds")
plt.title("Plot of Support threshold vs running times for different algorithms")

plt.legend()
plt.savefig(fig_name + ".png") #Change x with name given in the input script




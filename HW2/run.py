from cProfile import label
import sys
import time
import os
import matplotlib.pyplot as plt

filename1 = "converted_gspan.txt"
filename2 = "converted_fsg.txt"

minSup = [5, 10, 25, 50, 95]
algos = ["gSpan", "gaston", "fsg"]
times = {}

for algorithm in algos:
    times[algorithm] = []

for i in minSup:
    for algorithm in algos:
        if algorithm == "gSpan":
            run = "./gSpan-64 -f " + filename1 + " -s " + str(float(i/100)) + " -o"
        elif algorithm == "gaston":
            run = "./gaston " + str(64110*i/100.0) + " " + filename1 + " out.txt"
        else:
            run = "./fsg -s " + str(i) + " " + filename2

        start = time.time()
        os.system(run)
        end = time.time()
        times[algorithm].append(end - start)

print(times)
plt.plot(minSup, times["gaston"], label = "Gaston")
plt.plot(minSup, times["gSpan"], label = "gSpan")
plt.plot(minSup, times["fsg"], label = "FSG")
plt.title("Runtime vs minSup for different Algorithms")
plt.xlabel("minSup")
plt.ylabel("Runtime (in sec)")
plt.legend()
plt.savefig("runtime comparision.png")
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

dataset = sys.argv[1]
dimension = int(sys.argv[2])
pltname = sys.argv[3]

f = open(dataset, "r")
data = []
# x1 = []
# x2 = []
for x in f:
    s = x.split()
    temp = []
    for i in s:
        temp.append(float(i))
    data.append(temp)
    # x1.append(temp[0])
    # x2.append(temp[1])

# print(x1, x2)
# plt.scatter(x1, x2)
# plt.savefig("data.png")

# print(data)
k = []
variation = []
for i in range(1,16):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    k.append(i)
    variation.append(kmeans.inertia_)

# print(variation)
plt.plot(k, variation)
plt.title("Elbow plot for " + str(dimension) + "D data")
plt.xlabel("K (No of clusters)")
plt.ylabel("Variation within clusters (KMeans Inertia)")
plt.savefig(pltname)
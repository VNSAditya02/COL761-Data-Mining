Git Repo Link: https://github.com/RaviSriTejaKuriseti/Data-Mining-Assignments

Team Members:

1)KURISETI RAVI SRI TEJA - 2019CS10369
2)APPAKONDA ABHIJITH REDDY - 2019CS10330
3)BOMMAKANTI VENKATA NAGA SAI ADITYA - 2019CS50471

File Details:
apriori.cpp - Cpp file for Apriori Algorithm
fptree.cpp - Cpp file for FP-Tree Algorithm
plotting_script.py - Script to plot Graph of Time vs Support Threshold for Apriori and FP-Tree Algorithms
graph_apriori.cpp - Called while plotting, outputs time taken by Apriori Algorithm
graph_fptree.cpp - Called while plotting, outputs time taken by FP-Tree Algorithm

Contribution: All three of us worked on all parts equally

Contribution Percentage:

1)Ravi : 33
2)Abhijith : 33
3)Aditya : 33

Explanation of Results:

From the plot, we can observe that FP-Tree has taken less time when compared to Apriori Algorithm. FP Tree is better than Apriori Algorithm because in Apriori Algorithm we have to make disk accesses k time (where k is max size of frequent item sets) because we have to go through entire file for each item set size, whereas in FP-Tree Algorithm we just have to make 2 disk accesses (one for counting frequencies of one size item sets and other for building FP-Tree).

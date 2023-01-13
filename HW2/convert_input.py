import sys

filename = sys.argv[1]
algorithm = sys.argv[2]

f = open(filename, "r")

if algorithm == "gSpan" or algorithm == "gaston":
    output = open("converted_gspan.txt", "w")
    d = {}
    readVertices = False
    vertexId = 0
    numVertices = 0
    numGraphs = 0
    for x in f:
        x = x[:-1]
        if len(x) == 0:
            continue
        if x[0] == "#":
            output.write("t # " + str(numGraphs) + "\n")
            numGraphs += 1
            readVertices = False
        elif x.isnumeric():
            readVertices = not readVertices
            vertexId = 0
        elif readVertices:
            if x not in d:
                d[x] = numVertices
                numVertices += 1
            output.write("v " + str(vertexId) + " " + str(d[x]) + "\n")
            vertexId += 1
        else:
            output.write("e " + x + "\n")
    output.close()

else:
    output = open("converted_fsg.txt", "w")
    readVertices = False
    vertexId = 0
    numVertices = 0
    for x in f:
        x = x[:-1]
        if len(x) == 0:
            continue
        if x[0] == "#":
            output.write("t\n")
            readVertices = False
        elif x.isnumeric():
            readVertices = not readVertices
            vertexId = 0
        elif readVertices:
            output.write("v " + str(vertexId) + " " + x + "\n")
            vertexId += 1
        else:
            output.write("u " + x + "\n")
    output.close()




    

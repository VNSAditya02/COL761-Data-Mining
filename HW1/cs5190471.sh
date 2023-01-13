if [ $1 == "-apriori" ] 
then
   ./apriori $2 $3 $4

elif [ $1 == "-fptree" ] 
then
    ./fptree $2 $3 $4

elif [ $1 == "-plot" ] 
then
   ./graph_fptree $2
   ./graph_apriori $2
   python3 plotting_script.py $3
   rm fptree.txt
   rm apriori.txt
   
else
 echo "Argument not found"
 
fi

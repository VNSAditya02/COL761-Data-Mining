if [ $1 == "train" ] 
then
   echo "Training"
   python3 part_2a.py $2 $3 $4 $5 $6

elif [ $1 == "test" ] 
then
    echo "Testing"
    python3 part_2b.py $2 $3 $4 $5 $6
   
else
 echo "Argument not found"
 
fi

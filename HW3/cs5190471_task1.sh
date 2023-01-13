if [ $1 == "train" ] 
then
   echo "Training"
   python3 part_1a.py $2 $3 $4

elif [ $1 == "test" ] 
then
    echo "Testing"
    python3 part_1b.py $2 $3 $4
   
else
 echo "Argument not found"
 
fi

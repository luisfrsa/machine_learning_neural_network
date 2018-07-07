echo "Starting tensorflow";

layers="2,4,2"
epoca="500"
epoca2="1000"

echo "Running 01";
python neural_luis.py $layers $epoca2 16 mean_squared_logarithmic_error elu sgd 1 2 0 > result/out01.txt 2>&1;
echo "Running 02";
python neural_luis.py $layers $epoca2 16 mean_squared_logarithmic_error elu sgd 0 2 0 > result/out02.txt 2>&1;
echo "Running 03";
python neural_luis.py $layers $epoca 8 mean_squared_logarithmic_error elu sgd 0 3 0 > result/out03.txt 2>&1;
echo "Running 04";
python neural_luis.py $layers $epoca 4 mean_squared_logarithmic_error elu sgd 0 4 0 > result/out04.txt 2>&1;
echo "Running 05";
python neural_luis.py $layers $epoca 8 mean_squared_logarithmic_error elu sgd 0 5 0 > result/out05.txt 2>&1;
echo "Finished";


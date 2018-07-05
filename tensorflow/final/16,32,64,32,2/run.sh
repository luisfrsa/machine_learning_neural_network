python neural_luis.py 16,32,64,32,2 400 32 mean_squared_logarithmic_error selu sgd 1 2 0 > result/saida1.txt 2>&1;
echo "Second";
python neural_luis.py 16,32,64,32,2 800 15 mean_squared_logarithmic_error selu sgd 0 2 0 > result/saida2.txt 2>&1;
echo "Third";
python neural_luis.py 16,32,64,32,2 1000 5 mean_squared_logarithmic_error selu sgd 0 3 0 > result/saida3.txt 2>&1;
echo "Forth";
# python neural_luis.py 16,32,64,32,2 1000 1 mean_squared_logarithmic_error selu sgd 1 4 1 > result/saida4.txt 2>&1;
echo "Finish";
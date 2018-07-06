echo "Fiveth";
python neural_luis.py 2,2,2 1000 32 mean_squared_logarithmic_error selu sgd 0 4 0 > result/saida5.txt 2>&1;
echo "Sixth";
python neural_luis.py 2,2,2 1000 15 mean_squared_logarithmic_error selu sgd 0 4 0 > result/saida6.txt 2>&1;
echo "Seventh";
python neural_luis.pya ta 2,2,2 1000 5 mean_squared_logarithmic_error selu sgd 0 5 0 > result/saida7.txt 2>&1;
echo "Eighth";
python neural_luis.py 2,2,2 1000 1 mean_squared_logarithmic_error selu sgd 0 5 0 > result/saida8.txt 2>&1;
echo "Try";


layers="32,64,128,64,2"
epoca="1000"
epoca2="2000"


# echo "1";
# python neural_luis.py $layers $epoca2 64 mean_squared_logarithmic_error elu sgd 1 1 0 > result/saida01.txt 2>&1;
# python neural_luis.py $layers $epoca2 32 mean_squared_logarithmic_error elu sgd 0 1 0 > result/saida011.txt 2>&1;
# echo "2";
# python neural_luis.py $layers $epoca2 64 mean_squared_logarithmic_error elu sgd 0 2 0 > result/saida02.txt 2>&1;
# echo "3";
# python neural_luis.py $layers $epoca2 32 mean_squared_logarithmic_error elu sgd 0 2 0 > result/saida03.txt 2>&1;
# echo "4";
# python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 2 0 > result/saida04.txt 2>&1;
# echo "5";
# python neural_luis.py $layers $epoca2 64 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida05.txt 2>&1;
# echo "6";
# python neural_luis.py $layers $epoca2 32 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida06.txt 2>&1;
# echo "7";
# python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida07.txt 2>&1;
# echo "8";
# python neural_luis.py $layers $epoca2 64 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida08.txt 2>&1;
# echo "9";
# python neural_luis.py $layers $epoca2 32 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida09.txt 2>&1;
# echo "10";
# python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida010.txt 2>&1;
# echo "11";
# python neural_luis.py $layers $epoca2 64 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida011.txt 2>&1;
# echo "12";
# python neural_luis.py $layers $epoca2 32 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida012.txt 2>&1;
# echo "13";
# python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida013.txt 2>&1;
# echo "14";
# python neural_luis.py $layers $epoca2 64 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida014.txt 2>&1;
echo "15";
python neural_luis.py $layers $epoca2 32 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida015.txt 2>&1;
echo "16";
python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida016.txt 2>&1;
echo "17";
python neural_luis.py $layers $epoca2 64 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida017.txt 2>&1;
echo "18";
python neural_luis.py $layers $epoca2 32 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida018.txt 2>&1;
echo "19";
python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida019.txt 2>&1;

# echo "Pre";
# python neural_luis.py $layers $epoca 64 mean_squared_logarithmic_error elu sgd 0 1 0 > result/saida01.txt 2>&1;
# python neural_luis.py $layers $epoca 64 mean_squared_logarithmic_error elu sgd 0 2 0 > result/saida02.txt 2>&1;
# python neural_luis.py $layers $epoca 32 mean_squared_logarithmic_error elu sgd 0 2 0 > result/saida03.txt 2>&1;
# python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 2 0 > result/saida04.txt 2>&1;
# echo "First";
# python neural_luis.py $layers 500 32 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida1.txt 2>&1;
# echo "Second";
# python neural_luis.py $layers 500 16 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida2.txt 2>&1;
# echo "Third";
# python neural_luis.py $layers 500 10 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida3.txt 2>&1;
# echo "Forth";
# python neural_luis.py $layers 500 16 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida4.txt 2>&1;
# echo "fifth";
# python neural_luis.py $layers $epoca 10 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida5.txt 2>&1;
# echo "Sixth";
# python neural_luis.py $layers $epoca 5 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida6.txt 2>&1;
# echo "Seventh";
# python neural_luis.py $layers $epoca 10 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida7.txt 2>&1;
# echo "Eigth";
# python neural_luis.py $layers $epoca 5 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida8.txt 2>&1;
# echo "5";
# python neural_luis.py $layers $epoca 5 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida5.txt 2>&1;
# echo "6";
# python neural_luis.py $layers $epoca 3 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida6.txt 2>&1;
# echo "7";
# python neural_luis.py $layers $epoca 64 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida7.txt 2>&1;
# echo "8";
# python neural_luis.py $layers $epoca 32 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida8.txt 2>&1;
# echo "9";
# python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida9.txt 2>&1;
# echo "10";
# python neural_luis.py $layers $epoca 10 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida10.txt 2>&1;
# echo "11";
# python neural_luis.py $layers $epoca 5 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida11.txt 2>&1;
# echo "13";
# python neural_luis.py $layers $epoca 2 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida13.txt 2>&1;
# echo "14";
# # python neural_luis.py $layers $epoca 64 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida12.txt 2>&1;
# # echo "13";
# python neural_luis.py $layers $epoca 32 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida14.txt 2>&1;
# echo "15";
# python neural_luis.py $layers $epoca 16 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida14.txt 2>&1;
# echo "15";
# python neural_luis.py $layers $epoca 10 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida15.txt 2>&1;
# echo "16";
# python neural_luis.py $layers $epoca 5 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida16.txt 2>&1;
# echo "17";
# python neural_luis.py $layers $epoca 2 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida17.txt 2>&1;
# echo "16";
# python neural_luis.py $layers $epoca 5 mean_squared_logarithmic_error elu sgd 0 5 0 > result/saida16.txt 2>&1;
# echo "Finish";

echo "temp";
# python neural_luis.py $layers $epoca 32 mean_squared_logarithmic_error elu sgd 0 2 0 > result/saida22.txt 2>&1;
# python neural_luis.py $layers $epoca 32 mean_squared_logarithmic_error elu sgd 0 3 0 > result/saida23.txt 2>&1;
# python neural_luis.py $layers $epoca 32 mean_squared_logarithmic_error elu sgd 0 4 0 > result/saida24.txt 2>&1;
echo "Finish";

# python neural_luis.py $layers 400 32 mean_squared_logarithmic_error elu sgd 1 4 0 > result/saida_final.txt 2>&1;

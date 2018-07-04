python testes_luis.py 32,64,128,64,32,2 1000 1 mean_squared_error elu sgd 0 > result/saida_32,64,128,64,32,2_1000_5_mean_squared_error_elu_sgd_fim.txt 2>&1
python testes_luis.py 32,64,128,64,32,2 1000 1 mean_squared_error selu sgd 0 > result/saida_32,64,128,64,32,2_1000_5_mean_squared_error_selu_sgd_fim.txt 2>&1
python testes_luis.py 32,64,128,64,32,2 1000 1 mean_squared_logarithmic_error selu sgd 0 > result/saida_32,64,128,64,32,2_1000_5_mean_squared_logarithmic_error_selu_sgd_fim.txt 2>&1
python testes_luis.py 32,64,128,64,32,2 1000 1 squared_hinge selu sgd 0 > result/saida_32,64,128,64,32,2_1000_5_squared_hinge_selu_sgd_fim.txt 2>&1

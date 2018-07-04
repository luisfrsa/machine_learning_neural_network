declare -a PARAM_DENSE=("32,64,128,64,32,2")
# declare -a PARAM_DENSE=("32,64,32,2")
declare -a PARAM_EPOCHS=("1000")
declare -a PARAM_BATCH_SIZE=("5")
declare -a PARAM_LOSS=("mean_squared_logarithmic_error" "mean_squared_error" "squared_hinge")
declare -a PARAM_ACTIVATION=("elu" "selu" "crelu")
declare -a PARAM_OPTIMIZER=("sgd" "rms" "adadelta")
NEW=1

#python testes_luis.py 32,64,32,2 200 5 squared_hinge selu rms > result/saida_32,64,128,64,32,2_200_5_squared_hinge_selu_rms_fim.txt 2>&1
for PARAM_DENSE_VAR in "${PARAM_DENSE[@]}";
do
    for PARAM_EPOCHS_VAR in "${PARAM_EPOCHS[@]}";
    do
        for PARAM_BATCH_SIZE_VAR in "${PARAM_BATCH_SIZE[@]}";
        do
            for PARAM_LOSS_VAR in "${PARAM_LOSS[@]}";
            do
                for PARAM_ACTIVATION_VAR in "${PARAM_ACTIVATION[@]}";
                do
                    for PARAM_OPTIMIZER_VAR in "${PARAM_OPTIMIZER[@]}";
                    do
                        #python testes_luis.py $PARAM_DENSE_VAR $PARAM_EPOCHS_VAR $PARAM_BATCH_SIZE_VAR $PARAM_LOSS_VAR $PARAM_ACTIVATION_VAR $PARAM_OPTIMIZER_VAR $NEW > result/saida_${PARAM_DENSE_VAR}_${PARAM_EPOCHS_VAR}_${PARAM_BATCH_SIZE_VAR}_${PARAM_LOSS_VAR}_${PARAM_ACTIVATION_VAR}_${PARAM_OPTIMIZER_VAR}_fim.txt 2>&1
                        echo "python testes_luis.py $PARAM_DENSE_VAR $PARAM_EPOCHS_VAR $PARAM_BATCH_SIZE_VAR $PARAM_LOSS_VAR $PARAM_ACTIVATION_VAR $PARAM_OPTIMIZER_VAR $NEW > result/saida_${PARAM_DENSE_VAR}_${PARAM_EPOCHS_VAR}_${PARAM_BATCH_SIZE_VAR}_${PARAM_LOSS_VAR}_${PARAM_ACTIVATION_VAR}_${PARAM_OPTIMIZER_VAR}_fim.txt 2>&1"                        
                    done
                done
            done
        done
    done
done
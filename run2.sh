declare -a PARAM_DENSE=("32,64,128,64,32,2")
declare -a PARAM_EPOCHS=("200")
declare -a PARAM_BATCH_SIZE=("5")
declare -a PARAM_LOSS=("mean_squared_logarithmic_error" "mean_squared_error" "squared_hinge")
declare -a PARAM_ACTIVATION=("elu" "selu" "crelu")
declare -a PARAM_OPTIMIZER=("sgd" "rms" "adadelta")


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
                        #python teste_luis.py $PARAM_DENSE_VAR $PARAM_EPOCHS_VAR $PARAM_BATCH_SIZE_VAR $PARAM_LOSS_VAR $PARAM_ACTIVATION_VAR $PARAM_OPTIMIZER_VAR > result/saida_$PARAM_DENSE_VAR_$PARAM_EPOCHS_VAR_$PARAM_BATCH_SIZE_VAR_$PARAM_LOSS_VAR_$PARAM_ACTIVATION_VAR_$PARAM_OPTIMIZER_VAR_fim.txt
                        echo "python teste_luis.py $PARAM_DENSE_VAR $PARAM_EPOCHS_VAR $PARAM_BATCH_SIZE_VAR $PARAM_LOSS_VAR $PARAM_ACTIVATION_VAR $PARAM_OPTIMIZER_VAR > result/saida_$PARAM_DENSE_VAR_$PARAM_EPOCHS_VAR_$PARAM_BATCH_SIZE_VAR_$PARAM_LOSS_VAR_$PARAM_ACTIVATION_VAR_$PARAM_OPTIMIZER_VAR_fim.txt"                        
                    done
                done
            done
        done
    done
done
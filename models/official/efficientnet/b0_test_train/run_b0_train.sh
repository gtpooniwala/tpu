#!/bin/bash

# ####################### Executed on 2019-02-11 ##########################
# # Pruning 90%, lr 1.25e-3, Based on 'cudnn_gradual_final_125_80'
# tensorflow.sub ngpu=8 py=2 env="source /tool/ml/cuda9/.venv/settings_py2-tf18.bash" j='cudnn_gradual_prune_recomp_90_125e-5' ./train_ctc_tf_gradual_prune_recomp.sh ./exp/cudnn_gradual_prune_recomp_90_125e-5/ 90 '0.00125'
#
# # Pruning 90%, lr 5e-4, Based on 'cudnn_gradual_final_125_80'
# tensorflow.sub ngpu=8 py=2 env="source /tool/ml/cuda9/.venv/settings_py2-tf18.bash" j='cudnn_gradual_prune_recomp_90_125e-5' ./train_ctc_tf_gradual_prune_recomp.sh ./exp/cudnn_gradual_prune_recomp_90_5e-4/ 90 '0.0005'
#

###############################Execution for efficientNet(2019/06/28)##################################
# export MODEL=efficientnet-b0
#tensorflow.sub ngpu=1 py=2 env="source /tool/ml/cuda9/.venv/settings_py2-tf112-hub.bash" j='Train EfficientNet B0 Testrun' ./b0_train.sh
tensorflow.sub ngpu=1 m=srp100-0199 py=2 env="source /tool/ml/cuda9/.venv/settings_py2-tf112-hub.bash" j='Train EfficientNet B0 Testrun' ./b0_train.sh


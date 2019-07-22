#!/bin/bash

# ####################### Executed on 2019-02-11 ##########################
# # Pruning 90%, lr 1.25e-3, Based on 'cudnn_gradual_final_125_80'
# tensorflow.sub ngpu=8 py=2 env="source /tool/ml/cuda9/.venv/settings_py2-tf18.bash" j='cudnn_gradual_prune_recomp_90_125e-5' ./train_ctc_tf_gradual_prune_recomp.sh ./exp/cudnn_gradual_prune_recomp_90_125e-5/ 90 '0.00125'
#
# # Pruning 90%, lr 5e-4, Based on 'cudnn_gradual_final_125_80'
# tensorflow.sub ngpu=8 py=2 env="source /tool/ml/cuda9/.venv/settings_py2-tf18.bash" j='cudnn_gradual_prune_recomp_90_125e-5' ./train_ctc_tf_gradual_prune_recomp.sh ./exp/cudnn_gradual_prune_recomp_90_5e-4/ 90 '0.0005'
#

###############################Execution for efficientNet(2019/06/28)##################################
export MODEL=efficientnet-b0

# tensorflow.sub ngpu=8 py=2 env="source /tool/ml/cuda9/.venv/settings_py2-tf18.bash" j='EfficientNet B0 EvalTest' python main.py --use_tpu=false --data_dir=/dbstore/sr_odl/datasets/ILSVRC2012/TFRecords_Imagenet/ --model_dir=./B0_test --train_batch_size=128 --eval_batch_size=64 --log_step_count_steps=10 --steps_per_eval=10000 --iterations_per_loop=10000
cd ..
python eval_ckpt_main.py --model_name=$MODEL --ckpt_dir=$MODEL --example_img=panda.jpg --labels_map_file=labels_map.txt

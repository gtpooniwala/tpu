#!/bin/bash


###############################Execution for efficientNet(2019/06/28)##################################
# echo "Code running"
cd ..
export PYTHONPATH="$PYTHONPATH:./pretrained_models/efficientnet-b0/"
python main.py --use_tpu=false --data_dir=/dbstore/sr_odl/datasets/ILSVRC2012/TFRecords_ImageNet/ --model_dir=./b0_test_train/train_b0_1gpu --train_steps=1001 --train_batch_size=128 --eval_batch_size=64 --log_step_count_steps=10 --steps_per_eval=1000 --iterations_per_loop=10000
#train_steps, default=218949 whic is approx 350 epochs at batch size 2048

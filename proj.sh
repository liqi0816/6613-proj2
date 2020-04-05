#!/bin/sh

conda activate clvision-challenge
( 
>> base.log &)
( {
python main.py     --scenario="multi-task-nc" --epochs="2" --sub_dir="baseline"
python main_ewc.py --scenario="multi-task-nc" --epochs="2" --sub_dir="ewc50" --ewc_weight=50
python main_ewc.py --scenario="multi-task-nc" --epochs="2" --sub_dir="ewc100" --ewc_weight=100
python main_ewc.py --scenario="multi-task-nc" --epochs="2" --sub_dir="ewc300" --ewc_weight=300
python main_ewc.py --scenario="multi-task-nc" --epochs="2" --sub_dir="ewc500" --ewc_weight=500
python main_ewc.py --scenario="multi-task-nc" --epochs="2" --sub_dir="ewc1000" --ewc_weight=1000
python main_ewc.py --scenario="multi-task-nc" --epochs="2" --sub_dir="ewc3000" --ewc_weight=3000
} >> ewc.log &)

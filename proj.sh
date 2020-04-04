#!/bin/sh

conda activate clvision-challenge
( 
python main.py     --scenario="multi-task-nc" --epochs="5" --sub_dir="baseline"
>> base.log &)
( 
python main_ewc.py --scenario="multi-task-nc" --epochs="5" --sub_dir="ewc"
>> ewc.log &)

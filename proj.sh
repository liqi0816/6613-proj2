#!/bin/sh

conda activate clvision-challenge

set -x
function main_ewc {
    PYTHONUNBUFFERED=1 python main_ewc.py \
        --scenario="multi-task-nc" \
        --sub_dir="ewc-$ewc_weight-$ewc_explosion_multr_cap-$(date '+%m%d%H%M')" \
        --epochs="2" \
        --ewc_weight=$ewc_weight \
        --ewc_explosion_multr_cap=$ewc_explosion_multr_cap
}
function main_ewc_var_weight {
    for ((ewc_weight = 5; ewc_weight < 101; ewc_weight+=5 )); do
        ewc_weight=$ewc_weight main_ewc
    done
}
for ((ewc_explosion_multr_cap = 5; ewc_explosion_multr_cap < 20; ewc_explosion_multr_cap+=2 )); do
	ewc_explosion_multr_cap=$ewc_explosion_multr_cap main_ewc_var_weight
done >> "batch-exp-$(date '+%m%d%H%M').log"
# PYTHONUNBUFFERED=1 python main.py --scenario="multi-task-nc" --epochs="2" --sub_dir="baseline-$(date '+%m%d%H%M')"

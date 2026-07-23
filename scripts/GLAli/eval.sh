#!/bin/bash

# custom config
TRAINER=LocProto

CSC=False
CTP=end

DATA=data/
# DATASET=skin40
CFG=vit_b16_ep25

NCTX=16

T=1
kalign=10
# SHOTS=16
# MODEL_dir=$5
# Output_dir=$5

# SEED=2
for DATASET in ${1:-BTXRD}  # eurosat fgvc_aircraft stanford_cars skin40 ISIC
do
    for SHOTS in 16
    do
        for SEED in 1
        do
            CUDA_VISIBLE_DEVICES=0 python eval_ood_detection.py \
            --root ${DATA} \
            --in_dataset ${DATASET} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --seed ${SEED} \
            --output-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
            --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
            --load-epoch 200 \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --T ${T} \
            --use_refined \
            --kalign ${kalign} \
            DATASET.SUBSAMPLE_CLASSES base \
            DATASET.NUM_SHOTS ${SHOTS} \
            TEST.PER_CLASS_RESULT True \
            TEST.COMPUTE_CMAT True
        done
    done
done
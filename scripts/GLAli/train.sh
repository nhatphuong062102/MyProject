#!/bin/bash
# custom config
TRAINER=LocProto

DATA=data/
# DATASET=$1
DATASET=${1:-BTXRD}
# CFG=$3  # config file
CFG=vit_b16_ep25
# CTP=$4  # class token position (end or middle)
CTP=end
# NCTX=$5  # number of context tokens
NCTX=16
# SHOTS=$6  # number of shots (1, 2, 4, 8, 16)
# CSC=$7  # class-specific context (False or True)
CSC=False
lambda=0.25
topk=50
kalign=10

for SEED in 1
do
    for SHOTS in 16
    do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        # if [ -d "$DIR" ]; the6
        #     echo "Oops! The results exist at ${DIR} (so skip this job)"
        # else
        echo $PWD
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --lambda_value ${lambda} \
        --topk ${topk} \
        --kalign ${kalign} \
        --is_bonder True \
        --use_refined True \
        --is_dense True \
        --is_sc True \
        TRAINER.LOCOOP.N_CTX ${NCTX} \
        TRAINER.LOCOOP.CSC ${CSC} \
        TRAINER.LOCOOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base \
        #TRAIN.CHECKPOINT_FREQ 5 \
        #TEST.PER_CLASS_RESULT True \
    done
done
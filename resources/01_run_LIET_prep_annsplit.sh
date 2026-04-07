#!/bin/bash

LIET_DIR=/Users/hoto7260/LIET/LIET
INPUT_DIR=${LIET_DIR}/liet/tests
srr_list=${INPUT_DIR}/test_SRRs.txt
annfile=${INPUT_DIR}/LIET_input_1210.txt
BGS=${INPUT_DIR}/bedgraphs
WD=${LIET_DIR}/resources/LIET_tmp/
LIET_DIR=${LIET_DIR}/liet



## BGS_done
ann_name=Testing_LIET
bash LIET_prep_annsplit.sh ${srr_list} ${annfile} ${BGS} ${WD} "BGS_done" ${ann_name} "PAD=3000,3000" ${LIET_DIR}

## If you have a pad file
# bash LIET_prep_annsplit.sh ${srr_list} ${annfile} ${BGS} ${WD} "BGS_done" ${ann_name} "PAD_FILE=${PAD_FILE}" ${LIET_DIR}



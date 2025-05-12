#!/bin/bash


INPUT_DIR=/Users/hoto7260/LIET/LIET/liet/tests
srr_list=${INPUT_DIR}/test_SRRs.txt
annfile=${INPUT_DIR}/LIET_input_1210.txt
BGS=${INPUT_DIR}/bedgraphs
WD=LIET_tmp/
LIET_DIR=/Users/hoto7260/LIET/LIET/liet


## BGS_done, BAMS
ann_name=Testing_LIET
bash LIET_prep_annsplit.sh ${srr_list} ${annfile} ${BGS} ${WD} "BGS_done" ${ann_name} "PAD=3000,3000"

## If you have a pad file
# bash LIET_prep_annsplit.sh ${srr_list} ${annfile} ${BGS} ${WD} "BGS_done" ${ann_name} "PAD_FILE=${PAD_FILE}"

# ann_name=CUWA_FixedPadded_05.8.25
# bash 02_LIET_prep_annsplit.sh ${srr_list} ${annfile} ${BGS} ${WD} "BGS_done" ${ann_name} "PAD_FILE=3000,3000"



# INPUT_DIR=/scratch/Users/hoto7260/Resp_Env/Comb_UPM_WSP_ADP/LIET/LIET_input/
# BG_DIR=/scratch/Users/hoto7260/nextflow_out/Bidir/UPM_smAECs_07-23-24/bedgraphs/
# srr=sm36-ALI-D21_120UPM-2
# bg=${BG_DIR}/${srr}.sorted_3.bedGraph
# pos_bg=${INPUT_DIR}/bedgraphs/${srr}_3.pos.sorted.bedGraph
# neg_bg=${INPUT_DIR}/bedgraphs/${srr}_3.neg.sorted.bedGraph
# grep "-" $bg | sed -r 's/-//' > ${neg_bg}
# grep -v "-" $bg > ${pos_bg}

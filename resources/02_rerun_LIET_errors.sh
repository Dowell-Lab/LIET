#!/bin/bash

module load R/4.4.0

######################
# Edit the Following #
######################



OD=/scratch/Users/hoto7260/Resp_Env/Comb_UPM_WSP_ADP/LIET/
srr_list=(SRR13772349 SRR13772350 SRR13772351 SRR13772352)
LIET_DIR=/Users/hoto7260/LIET/Jacob_LIET2/LIET/liet/
PAD_FILE=${INPUT_DIR}/Pad_file_CUWA_8500SIG_05.8.25.txt
PAD="PAD_FILE=${PAD_FILE}"
BGS=/scratch/Users/hoto7260/Resp_Env/Comb_UPM_WSP_ADP/LIET/LIET_input/bedgraphs
## if no BG directory then
#BGS=${OD}/mapped/bedgraphs/


############################
# Defining terms & folders #
############################
LIET_EXE=${LIET_DIR}/liet_exe_mp.py
ANN_DIR=${OD}/LIET/ANN_DIR_TMP

EO_DIR=${OD}/e_and_o
RUN_LIET_DIR=${OD}/LIET/run-liet/



for srr in "${srr_list[@]}"; do
echo "============================= $srr ========================="
# create a file holding the annotations to redo
ann_prefix=redo_annot
pos_bg=${BGS}/${srr}_3.pos.sorted.bedGraph
neg_bg=${BGS}/${srr}_3.neg.sorted.bedGraph
OG_RESULTS_DIR=${OD}/LIET/LIET_results/${srr}_EMG
RESULTS_DIR=${OD}/LIET/LIET_results/${srr}_EMG_err
CONFIG_DIR=${OD}/LIET/configs/${srr}/redo_errors
mkdir ${RESULTS_DIR}
mkdir ${CONFIG_DIR}
##############################
# GET THE NEEDED ANNOTATIONS #
##############################

# # make the file brand new to hold regions
# if test -f ${ANN_DIR}/${srr}_EMG_redo_annots.bed ; then
#   echo "Previous error file existed. Overwriting."
#   rm ${ANN_DIR}/${srr}_EMG_redo_annots.bed
# fi

# # for each file in the output
# for out_file in ${OG_RESULTS_DIR}/[0-9]*.liet; do
#     #echo "Out file is", ${out_file}
#     # copy the lines indicating errors to a model error file
#     grep error ${out_file} > ${OG_RESULTS_DIR}/model_error.txt
#     # now remove the error components to only leave regions
#     sed -i -e 's/: model error//g' ${OG_RESULTS_DIR}/model_error.txt
#     sed -i -e 's/: fitting error//g' ${OG_RESULTS_DIR}/model_error.txt
#     # add the annotations of these features to the redo file
#     # get the annot file via the prefix (remove .liet)
#     prefix=${out_file##*/}
#     prefix="${prefix%.*}"
#     # remove Full from prefix
#     prefix="${prefix%_EMG}"
#     #echo "Getting errors for prefix $prefix"
#     awk 'NR==FNR {a[$0];next} $4 in a' ${OG_RESULTS_DIR}/model_error.txt ${ANN_DIR}/${prefix}.txt >> ${ANN_DIR}/${srr}_EMG_redo_annots.bed
# done

# echo "Number of features that have to be redone due to a model error"
# wc -l ${ANN_DIR}/${srr}_EMG_redo_annots.bed

# line_count=$(wc -l < "${ANN_DIR}/${srr}_EMG_redo_annots.bed")

# if [ "$line_count" -eq 0 ]; then
#     echo "Skipping ${srr} because there were no errors."
#     continue
# fi

# # splitting up the annotation files
# Rscript ${LIET_DIR}/get_split_annotations.r ${ANN_DIR}/${srr}_EMG_redo_annots.bed ${ANN_DIR} ${ann_prefix}_${srr}_EMG

###########################
# CREATE THE CONFIG FILES #
###########################
for ann_subset in ${ANN_DIR}/*_${ann_prefix}_${srr}_EMG.txt; do
        ann_subset_prefix=${ann_subset##*/}
        ann_subset_prefix="${ann_subset_prefix%.*}"
        echo "PREFIX" ${ann_subset_prefix}
        # Create the config files

        # MAKE CONFIG
        CONFIG_FILE="${CONFIG_DIR}/${ann_subset_prefix}_.liet.config"
        echo "#### writing the config file at" ${CONFIG_FILE} 
        # Start writing configuration parameters to the config file
        echo "[FILES]" > "$CONFIG_FILE"
        echo "ANNOTATION=${ann_subset}" >> "$CONFIG_FILE"
        echo "BEDGRAPH_POS=${pos_bg}" >> "$CONFIG_FILE"
        echo "BEDGRAPH_NEG=${neg_bg}" >> "$CONFIG_FILE"
        echo "RESULTS=${RESULTS_DIR}/${ann_subset_prefix}.liet" >> "$CONFIG_FILE"
        
        # if pad file, add here
        if [[ $PAD == *"PAD_FILE"* ]]; then
        echo ${PAD} >> "$CONFIG_FILE"
        fi
        echo "" >> "$CONFIG_FILE"
        echo "[MODEL]" >> "$CONFIG_FILE"
        echo "ANTISENSE=True" >> "$CONFIG_FILE"
        echo "BACKGROUND=True" >> "$CONFIG_FILE"
        echo "FRACPRIORS=False" >> "$CONFIG_FILE"
        echo "ET_sense=False" >> "$CONFIG_FILE"
        echo "ET_antisense=False" >> "$CONFIG_FILE"
        echo "" >> "$CONFIG_FILE"

        echo "[PRIORS]" >> "$CONFIG_FILE"
        echo "mL=dist:normal,mu:0,sigma:500" >> "$CONFIG_FILE"
        echo "sL=dist:exponential,tau:50,offset:15" >> "$CONFIG_FILE" 
        echo "tI=dist:exponential,tau:30,offset:20" >> "$CONFIG_FILE" 
        echo "mT=dist:exponential,tau:450,offset:100" >> "$CONFIG_FILE" 
        echo "sT=dist:exponential,tau:40,offset:0" >> "$CONFIG_FILE" 
        echo "w=dist:dirichlet,alpha_LI:1,alpha_E:1,alpha_T:1,alpha_B:1" >> "$CONFIG_FILE" 
        echo "mL_a=dist:normal,mu:0,sigma:500" >> "$CONFIG_FILE"
        echo "sL_a=dist:exponential,tau:50,offset:15" >> "$CONFIG_FILE" 
        echo "tI_a=dist:exponential,tau:30,offset:20" >> "$CONFIG_FILE" 
        echo "mT_a=dist:exponential,tau:450,offset:100" >> "$CONFIG_FILE" 
        echo "sT_a=dist:exponential,tau:40,offset:0" >> "$CONFIG_FILE" 
        echo "" >> "$CONFIG_FILE"


        echo "[DATA_PROC]" >> "$CONFIG_FILE"
        echo "RANGE_SHIFT=True" >> "$CONFIG_FILE"
        # if pad file then the default padding can be 2000,2000
        if [[ $PAD == *"PAD_FILE"* ]]; then
        echo "PAD=2000,2000" >> "$CONFIG_FILE"
        else
        echo ${PAD} >> "$CONFIG_FILE"
        fi
        echo "" >> "$CONFIG_FILE" 
        echo "[FIT]" >> "$CONFIG_FILE"
        echo "ITERATIONS=50000" >> "$CONFIG_FILE" 
        echo "LEARNING_RATE=0.05" >> "$CONFIG_FILE" 
        echo "METHOD=advi" >> "$CONFIG_FILE"
        echo "OPTIMIZER=adamax" >> "$CONFIG_FILE" 
        echo "MEANFIELD=True" >> "$CONFIG_FILE" 
        echo "TOLERANCE=1e-5" >> "$CONFIG_FILE" 
        echo "" >> "$CONFIG_FILE"

        echo "[RESULTS]" >> "$CONFIG_FILE" 
        echo "SAMPLES=50000" >> "$CONFIG_FILE" 
        echo "MEAN=True" >> "$CONFIG_FILE" 
        echo "MODE=False" >> "$CONFIG_FILE" 
        echo "MEDIAN=False" >> "$CONFIG_FILE" 
        echo "STDEV=True" >> "$CONFIG_FILE" 
        echo "SKEW=False" >> "$CONFIG_FILE" 
        echo "PERCENTILES=True" >> "$CONFIG_FILE" 
        echo "PDF=False" >> "$CONFIG_FILE" 
done
        
        
###########################
# CREATE THE SBATCH SCRIPT #
###########################

SBATCH_FILE=${RUN_LIET_DIR}/${srr}_EMG_error.sbatch
    echo "writing the sbatch script file ${SBATCH_FILE}"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=$srr"
        echo "#SBATCH --mail-type=FAIL,END"
        echo "#SBATCH --mail-user=hoto7260@colorado.edu"
        echo "#SBATCH -p highmem #specify long"
        echo "#SBATCH -N 1"
        echo "#SBATCH -c 64"
        echo "#SBATCH --mem=45gb"
        echo "#SBATCH --time=80:00:00"
        echo "#SBATCH --output=${EO_DIR}/run_LIET_%j.out"
        echo "#SBATCH --error=${EO_DIR}/run_LIET_%j.err"
        echo ""
        echo "### Clear modules and load conda environment"
        echo "module purge"
        echo "source /Users/hoto7260/miniconda3/bin/activate"
        echo "conda activate pymc5"
        echo ""
        echo "### LIET executable"
        echo "LIET_EXE='${LIET_EXE}'"
        echo "srr=${srr}"
        echo "CONFIG_DIR=${CONFIG_DIR}"s
        echo "RESULTS_DIR=${RESULTS_DIR}"
        echo "COMPILE_DIR=/scratch/Users/hoto7260/.pytensor_${srr}_EMG"
        echo "## set the pytensor compilation directory to be specific to the SRR being run"
        echo 'export PYTENSOR_FLAGS="base_compiledir=${COMPILE_DIR},cmodule__warn_no_version=True,cmodule__compilation_warning=True"'
        echo 'echo "Using pytensor flags: ${PYTENSOR_FLAGS}"'
        echo 'pytensor-cache clear'
        echo 'pytensor-cache'
        
        echo "for CONFIG_FILE in ${CONFIG_DIR}/*.config; do"
        echo 'printf "\n############ Using config file ${CONFIG_FILE}\n"'
        echo 'date'
        echo 'python ${LIET_EXE} -c ${CONFIG_FILE}'
        echo 'wait'
        echo 'echo "Previous job done"'
        echo 'date'
        echo 'echo "only when run is done, clear the pytensor files"'
        echo 'pytensor-cache purge'
        echo 'done'
        echo 'echo "Combining files"'
        echo 'rm -rf ${COMPILE_DIR}'
        echo 'cat ${RESULTS_DIR}/*.liet > ${RESULTS_DIR}/${srr}_EMG.liet'
        echo 'cat ${RESULTS_DIR}/*.liet.err > ${RESULTS_DIR}/${srr}_EMG.liet.err'
        echo 'cat ${RESULTS_DIR}/*.liet.log > ${RESULTS_DIR}/${srr}_EMG.liet.log'
        echo 'echo "DONE!"'
    } > "$SBATCH_FILE"




done
echo "DONE!"


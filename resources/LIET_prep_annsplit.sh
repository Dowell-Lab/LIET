#!/bin/bash
module load samtools/1.8
module load bedtools/2.28.0

source activate /Users/hoto7260/miniconda3/envs/Rjupyter
## conda environment with r-data.table installed
#conda list
#${srr_list} ${annfile} ${BAMS} ${WD} "CLEAN_BAMS" ${ann_name} "PAD_FILE=${PAD_FILE}"
srr_list=$1
annfile=$2
INPUT_DIR=$3
WD=$4
input=$5
ann_prefix=$6
# PAD must already be in teh format for the file
#. e.g. PAD=2000,2000
# or e.g. PAD_FILE=/scrat/ch//////
PAD=$7
LIET_DIR=$8

echo "one is srr_list" $srr_list
echo "two is annfile" $annfile
echo "three is INPUT_DIR (e.g. Bedgraphs)" $INPUT_DIR
echo "four is WD" $WD
echo "five is input type" $input
echo "six is ann_prefix" $ann_prefix
echo "seven is pad" $PAD
echo "eight is LIET direcotry" $LIET_DIR


# :' This script takes a text file of SRRs, the annotation file used for running LIET, the input directory, and an \
# output directory to use for LIET. \

# It will get the 3 prime stranded bedgraphs for LIET and write the CONFIG and SBATCH Files in the given WD

### EDIT TOWARDS THE CORRECT LIET PATH
LIET_EXE=${LIET_DIR}/liet_exe_mp.py

## CONFIG, RESULTS, E_AND_O
EO_DIR=${WD}/e_and_o
CONFIG_DIR=${WD}/LIET/configs
RESULTS_DIR=${WD}/LIET/LIET_results
RUN_LIET_DIR=${WD}/LIET/run-liet
mkdir -p ${EO_DIR}
mkdir -p ${CONFIG_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${RUN_LIET_DIR}
ANN_DIR_TMP=${WD}/LIET/ANN_DIR_TMP
mkdir -p ${ANN_DIR_TMP}
## MAPPED INPUTS
NEW_BAM_DIR=${WD}/mapped/bams
BG_DIR=${WD}/mapped/bedgraphs
mkdir -p ${NEW_BAM_DIR}
mkdir -p ${BG_DIR}

chroms=${LIET_DIR}/chroms.txt
chrom_sizes=/scratch/Shares/dowell/genomes/hg38/hg38.chrom.sizes

echo "Using annfile ${annfile}"
echo "Using Input in ${INPUT_DIR}"
echo "Input format is ${input}"
echo "Using output directory ${WD}"
echo "Using PAD ${PAD}"
echo "List of SRRs is" ${srr_list}


###################################################
## SPLIT THE ANNOTATION FILE INTO 1.45K SEGMENTS ###
###################################################
echo "############################ Splitting up the annotation files for running ############################"
# splitting up the annotation files
Rscript ${LIET_DIR}/get_split_annotations.r ${annfile} ${ANN_DIR_TMP} ${ann_prefix}

wc -l ${ANN_DIR_TMP}/*

## FOR EACH SRR, GET INPUT AND CONFIG FILES READY
while read srr; do
    echo "############################# Prepping for ${srr} ########################"
    ###################################
    ## GET STRANDED BEDGRAPHS READY ###
    ###################################
    if [ ${input} == "BAMS" ]; then
        echo "Getting clean bam file"
        bam=${INPUT_DIR}/${srr}.sorted.bam
        new_bam=${NEW_BAM_DIR}/${srr}.mmfilt.sorted.bam
        # Get clean version of bam
        samtools view -@ 16 -h -q 1 ${bam} | \
           grep -P '(NH:i:1|^@)' | \
           samtools view -h -b > ${new_bam}
        samtools index ${new_bam} ${new_bam}.bai
        
        # Get the 3' Bedgraphs
        echo "Getting the 3' stranded bedgraphs"
        pos_bg=${BG_DIR}/${srr}_3.pos.sorted.bedGraph
        neg_bg=${BG_DIR}/${srr}_3.neg.sorted.bedGraph
        # positive strand
        genomeCoverageBed \
            -bg \
            -strand + \
            -3 \
            -ibam ${new_bam} \
            > ${pos_bg}_nons
        # negative strand
        genomeCoverageBed \
            -bg \
            -strand - \
            -3 \
            -ibam ${new_bam} \
            > ${neg_bg}_nons
        # filter bedgraphs to only include primary chromosomes
        awk 'NR==FNR {a[$1];next}  $1 in a' ${chroms} ${pos_bg}_nons > ${pos_bg}_nons2
        awk 'NR==FNR {a[$1];next}  $1 in a' ${chroms} ${neg_bg}_nons > ${neg_bg}_nons2

        # Sort the bedgraphs
        bedtools sort -i ${neg_bg}_nons2 > ${neg_bg}
        bedtools sort -i ${pos_bg}_nons2 > ${pos_bg}
        rm ${pos_bg}_nons
        rm ${neg_bg}_nons
        rm ${pos_bg}_nons2
        rm ${neg_bg}_nons2

    elif [ ${input} == "CLEAN_BAMS" ]; then
        new_bam=${INPUT_DIR}/${srr}.mmfilt.sorted.bam
        # Get the 3' Bedgraphs
        echo "Getting the 3' stranded bedgraphs from CLEAN BAMS"
        pos_bg=${BG_DIR}/${srr}_3.pos.sorted.bedGraph
        neg_bg=${BG_DIR}/${srr}_3.neg.sorted.bedGraph
        # positive strand
        genomeCoverageBed \
            -bg \
            -strand + \
            -3 \
            -ibam ${new_bam} \
            > ${pos_bg}_nons
        # negative strand
        genomeCoverageBed \
            -bg \
            -strand - \
            -3 \
            -ibam ${new_bam} \
            > ${neg_bg}_nons
        # filter bedgraphs to only include primary chromosomes
        awk 'NR==FNR {a[$1];next}  $1 in a' ${chroms} ${pos_bg}_nons > ${pos_bg}_nons2
        awk 'NR==FNR {a[$1];next}  $1 in a' ${chroms} ${neg_bg}_nons > ${neg_bg}_nons2

        # Sort the bedgraphs
        bedtools sort -i ${neg_bg}_nons2 > ${neg_bg}
        bedtools sort -i ${pos_bg}_nons2 > ${pos_bg}
        rm ${pos_bg}_nons
        rm ${neg_bg}_nons
        rm ${pos_bg}_nons2
        rm ${neg_bg}_nons2
        wc -l ${pos_bg}
        wc -l ${neg_bg}
    
    elif [ ${input} == "BGS" ]; then
        # Get the 3' Bedgraphs
        echo "Getting the 3' stranded bedgraphs from UNION BEDGRAPHS"
        pos_bg=${BG_DIR}/${srr}_3.pos.sorted.bedGraph
        neg_bg=${BG_DIR}/${srr}_3.neg.sorted.bedGraph
        bg=${INPUT_DIR}/${srr}.sorted_3.bedGraph
        ##get the negative values and make positive (negative strand)
        grep "-" $bg | sed -r 's/-//' > ${neg_bg}
        ##get the non-negative values for positive strand
        grep -v "-" $bg > ${pos_bg}
        ##Double check they have the write numbers
        echo "Original #"
        wc -l $bg
        echo "Positive & Negative"
        wc -l ${pos_bg}
        wc -l ${neg_bg}
    elif [ ${input} == "BGS_done" ]; then
        echo "USING PROVIDED 3' stranded bedgraphs"
        pos_bg=${INPUT_DIR}/${srr}_3.pos.sorted.bedGraph
        neg_bg=${INPUT_DIR}/${srr}_3.neg.sorted.bedGraph
        echo "Positive & Negative"
        wc -l ${pos_bg}
        wc -l ${neg_bg}
    else
        echo "ERROR -- need BGS or BAMS noted as input"
        exit 125
    fi
    
    
    #################################
    ## MAKE CONFIG & SBATCH FILES ###
    mkdir -p ${CONFIG_DIR}/${srr}
    mkdir -p ${RUN_LIET_DIR}
    mkdir -p ${RESULTS_DIR}/${srr}_EMG
    for ann_subset in ${ANN_DIR_TMP}/*_${ann_prefix}.txt; do
        ann_subset_prefix=${ann_subset##*/}
        ann_subset_prefix="${ann_subset_prefix%.*}"
        
        ## MAKE CONFIG
        EMG_CONFIG_FILE="${CONFIG_DIR}/${srr}/${ann_subset_prefix}_EMG.liet.config"
        echo "#### writing the config file at ${EMG_CONFIG_FILE}"
        # Start writing configuration parameters to the config file
        echo "[FILES]" > "$EMG_CONFIG_FILE"
        echo "ANNOTATION=${ann_subset}" >> "$EMG_CONFIG_FILE"
        echo "BEDGRAPH_POS=${pos_bg}" >> "$EMG_CONFIG_FILE"
        echo "BEDGRAPH_NEG=${neg_bg}" >> "$EMG_CONFIG_FILE"
        echo "RESULTS=${RESULTS_DIR}/${srr}_EMG/${ann_subset_prefix}_EMG.liet" >> "$EMG_CONFIG_FILE"
        # if pad file, add here
        if [[ $PAD == *"PAD_FILE"* ]]; then
        echo ${PAD} >> "$EMG_CONFIG_FILE"
        fi
        echo "" >> "$EMG_CONFIG_FILE"
        echo "[MODEL]" >> "$EMG_CONFIG_FILE"
        echo "ANTISENSE=True" >> "$EMG_CONFIG_FILE"
        echo "BACKGROUND=True" >> "$EMG_CONFIG_FILE"
        echo "FRACPRIORS=False" >> "$EMG_CONFIG_FILE"
        echo "ET_sense=False" >> "$EMG_CONFIG_FILE"
        echo "ET_antisense=False" >> "$EMG_CONFIG_FILE"
        echo "" >> "$EMG_CONFIG_FILE"

        echo "[PRIORS]" >> "$EMG_CONFIG_FILE"
        echo "mL=dist:normal,mu:0,sigma:500" >> "$EMG_CONFIG_FILE"
        echo "sL=dist:exponential,tau:50,offset:15" >> "$EMG_CONFIG_FILE"
        echo "tI=dist:exponential,tau:30,offset:20" >> "$EMG_CONFIG_FILE"
        echo "mT=dist:exponential,tau:450,offset:100" >> "$EMG_CONFIG_FILE"
        echo "sT=dist:exponential,tau:40,offset:0" >> "$EMG_CONFIG_FILE"
        echo "w=dist:dirichlet,alpha_LI:1,alpha_E:1,alpha_T:1,alpha_B:1" >> "$EMG_CONFIG_FILE"
        echo "mL_a=dist:normal,mu:0,sigma:500" >> "$EMG_CONFIG_FILE"
        echo "sL_a=dist:exponential,tau:50,offset:15" >> "$EMG_CONFIG_FILE"
        echo "tI_a=dist:exponential,tau:30,offset:20" >> "$EMG_CONFIG_FILE"
        echo "mT_a=dist:exponential,tau:450,offset:100" >> "$EMG_CONFIG_FILE"
        echo "sT_a=dist:exponential,tau:40,offset:0" >> "$EMG_CONFIG_FILE"
        echo "" >> "$EMG_CONFIG_FILE"

        echo "[DATA_PROC]" >> "$EMG_CONFIG_FILE"
        echo "RANGE_SHIFT=True" >> "$EMG_CONFIG_FILE"
        # if pad file then the default padding can be 2000,2000
        if [[ $PAD == *"PAD_FILE"* ]]; then
        echo "PAD=3000,3000" >> "$EMG_CONFIG_FILE"
        else
        echo ${PAD} >> "$EMG_CONFIG_FILE"
        fi
        echo "" >> "$EMG_CONFIG_FILE"

        echo "[FIT]" >> "$EMG_CONFIG_FILE"
        echo "ITERATIONS=50000" >> "$EMG_CONFIG_FILE"
        echo "LEARNING_RATE=0.05" >> "$EMG_CONFIG_FILE"
        echo "METHOD=advi" >> "$EMG_CONFIG_FILE"
        echo "OPTIMIZER=adamax" >> "$EMG_CONFIG_FILE"
        echo "MEANFIELD=True" >> "$EMG_CONFIG_FILE"
        echo "TOLERANCE=1e-5" >> "$EMG_CONFIG_FILE"
        echo "" >> "$EMG_CONFIG_FILE"

        echo "[RESULTS]" >> "$EMG_CONFIG_FILE"
        echo "SAMPLES=50000" >> "$EMG_CONFIG_FILE"
        echo "MEAN=True" >> "$EMG_CONFIG_FILE"
        echo "MODE=False" >> "$EMG_CONFIG_FILE"
        echo "MEDIAN=False" >> "$EMG_CONFIG_FILE"
        echo "STDEV=True" >> "$EMG_CONFIG_FILE"
        echo "SKEW=False" >> "$EMG_CONFIG_FILE"
        echo "PERCENTILES=True" >> "$EMG_CONFIG_FILE"
        echo "PDF=False" >> "$EMG_CONFIG_FILE"
        
    done
    #####################################
    ## MAKE SBATCH FILES for each SRR ###
    EMG_SBATCH_FILE=${RUN_LIET_DIR}/${srr}_EMG.sbatch
    echo "writing the sbatch script file ${EMG_SBATCH_FILE}"
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
        echo "CONFIG_DIR=${CONFIG_DIR}"
        echo "RESULTS_DIR=${RESULTS_DIR}/${srr}_EMG"
        echo "COMPILE_DIR=/scratch/Users/hoto7260/.pytensor_${srr}_EMG"
        echo "## set the pytensor compilation directory to be specific to the SRR being run"
        echo 'export PYTENSOR_FLAGS="base_compiledir=${COMPILE_DIR},cmodule__warn_no_version=True,cmodule__compilation_warning=True"'
        echo 'echo "Using pytensor flags: ${PYTENSOR_FLAGS}"'
        echo 'pytensor-cache clear'
        echo 'pytensor-cache'
        
         echo "for ann_subset in ${ANN_DIR_TMP}/*_${ann_prefix}.txt; do"
        echo 'ann_subset_prefix=${ann_subset##*/}'
        echo 'ann_subset_prefix="${ann_subset_prefix%.*}"'
        echo 'echo ${ann_subset_prefix}'
        echo 'CONFIG_FILE=${CONFIG_DIR}/${srr}/${ann_subset_prefix}_EMG.liet.config'
        echo 'printf "\n############ Using config file ${CONFIG_FILE}\n"'
        echo 'date'
        echo 'python ${LIET_EXE} -c ${CONFIG_FILE}'
        echo 'wait'
        echo 'echo "Previous job done"'
        echo 'date'
        echo 'echo "only when run is done, clear the pytensor files"'
        echo 'pytensor-cache purge'
        echo 'done'
        echo 'rm -rf ${COMPILE_DIR}'
        echo 'echo "Combining files"'
        echo 'cat ${RESULTS_DIR}/*.liet > ${RESULTS_DIR}/${srr}_EMG.liet'
        echo 'cat ${RESULTS_DIR}/*.liet.err > ${RESULTS_DIR}/${srr}_EMG.liet.err'
        echo 'cat ${RESULTS_DIR}/*.liet.log > ${RESULTS_DIR}/${srr}_EMG.liet.log'
        echo 'echo "DONE!"'
        
    } > "$EMG_SBATCH_FILE"
     
    
done < "$srr_list"
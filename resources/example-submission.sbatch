#!/bin/bash
#SBATCH --job-name=<YOUR-JOB-NAME-HERE>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<YOUR-EMAIL-HERE>
#SBATCH -p highmem
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=45gb
#SBATCH --time=24:00:00
#SBATCH --output=<PATH-TO-YOUR-OUTPUT-HERE>
#SBATCH --error=<PATH-TO-YOUR-ERROR-HERE>

### Clear modules and load conda environment
module purge

# Activate your conda environment. 
# Note that the path below is an example and may not be how your envrionment is set up.
source /Users/<YOUR-USERNAME-HERE>/miniconda3/bin/activate

conda activate <YOUR-ENV-NAME>

### LIET executable 

# Note that the path below is an example and may not be where your executable is located.
# It should be located where you cloned the LIET respository.
LIET_EXE='/Users/<YOUR-USERNAME-HERE>/LIET/liet/liet_exe_mp.py'
printf "\n######################LIET EXE##########################\n"
cat $LIET_EXE
printf "\n########################################################\n"

# Make a directory for all of your PyMC temp files to be output to.
mkdir /scratch/Users/<YOUR-USERNAME-HERE>/liet-pytensor-temp-dirs/1
export PYTENSOR_FLAGS="base_compiledir=/scratch/Users/<YOUR-USERNAME-HERE>/liet-pytensor-temp-dirs/1"

# Print the compiledir base for confirmation
echo "Compiledir base is set to: $PYTENSOR_FLAGS"

config_file='<PATH-TO-YOUR-CONFIG-DIR>/config/<YOUR-SAMPLE-NAME>.liet.config'
echo "Processing file: $config_file"
python $LIET_EXE -c $config_file

# Delete the PyMC temp dir
rm -r /scratch/Users/<YOUR-USERNAME-HERE>/liet-pytensor-temp-dirs/1

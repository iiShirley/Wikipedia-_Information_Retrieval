#!/bin/bash
#SBATCH --job-name=problem6_indexing
#SBATCH --account=si650f25s101_class
#SBATCH --partition=standard
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=problem6_output.txt

# The application(s) to execute along with its input arguments and options:
/bin/hostname
module load python3.11-anaconda/2024.02
cd /home/shirleyi/HW1/HW1_export/hw1_starter_code
python problem6.py
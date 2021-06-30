#!/bin/bash 
#SBATCH --partition=long-serial 
#SBATCH --output=%j.out 
#SBATCH --error=%j.err
#SBATCH --time=48:00

conda activate nemo-medusa-cssm
python -u ./extract.py ./nemo-medusa-WIO.nc --minlon=20 --maxlon=78 --minlat=-38 --maxlat=25

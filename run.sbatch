#!/bin/bash
#SBATCH --partition=par-multi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --no-kill
##SBATCH --cpus-per-task 16
#SBATCH --output=%j.out 
#SBATCH --error=%j.err
#SBATCH --time=24:00:00
##SBATCH --mail-type=ALL
##SBATCH --mail-user=jorn@bolding-bruggeman.com

conda activate nemo-medusa-cssm

# Generate secret for Parallel Python authentication between server and client
secret=`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32`

# Pick IP port for Parallel Python communication (IANA: use 49152 to 65535 for dynamic/private ports)
port=`python -c 'import random; print(random.randint(49152,65535))'`

name=nemo-medusa-WIO

# Prevent numpy from using multiple threads (only if it has been built against Intel MKL)
export MKL_NUM_THREADS=1

# run the servers
srun ppserver.py -t 3600 -s $secret -p $port -r &

# launch the parallel processing
mkdir -p results
python -u run.py ${name}.nc results/${name} --ppservers $SLURM_JOB_NODELIST:$port --ncpus 0 --secret=$secret --debug

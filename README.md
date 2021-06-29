# NEMO-MEDUSA-CSSM

This repository contains scripts for simulation with the Community Size Spectrum Model, driven by the 1/4 degree "ROAM" projections from NEMO-MEDUSA.

## Running on JASMIN

### Set up

Inside the root of the local repository directory:

```
module load jaspy
conda env create -f environment.yml
conda activate nemo-ersem-cssm
source install
```

### Extracting NEMO-MEDUSA outputs

This is a serial job that can take over 24 hours.
It is therefore done on the long queue. It thus needs to be submitted to the SLURM scheduler. For this purpose, the `extract.slurm` job submission script is provided. Use it like this:

```
sbatch extract.slurm
```

Note that inside this script, the region to process is hard-coded with arguments `--minlon`, `--maxlon`, `--minlat`, `--maxlat`, as is the name of the output file (the unnamed argument to `extract.py`). If you would want to operate on a different region, either modify the script in-place, or create a copy and modify that.

### Running the Community Size Spectrum Model




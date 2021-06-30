# NEMO-MEDUSA-CSSM

This repository contains scripts for simulation with the Community Size Spectrum Model, driven by the 1/4 degree "ROAM" projections from NEMO-MEDUSA.

## Running on JASMIN

### Set up

After logging into one of JASMIN's "sci" machines (e.g., `sci2.jasmin.ac.uk`), first try running `conda`.
If you have previouly used [JASMIN's Python environment](https://help.jasmin.ac.uk/article/4729-jaspy-envs), that should work.
Otherwise, if you get "command not found", do the following:

```
module load jaspy
conda init bash
```

Then log out and back in. Now `conda` should work.

To set up the Python environment we will use for processing, first get a local copy of this repository:

```
git clone --recurse-submodules https://github.com/BoldingBruggeman/nemo-medusa-cssm.git
```

Move into the newly created `nemo-medusa-cssm` directory and do:

```
conda env create -f environment.yml
conda activate nemo-medusa-cssm
source ./install
```

This sets up a new, isolated Python environment. Note that this can later be removed if needed with `conda env remove nemo-medusa-cssm`.

### Extracting NEMO-MEDUSA outputs

This is a serial job that can take over 24 hours.
It is therefore done on [the `long-serial` queue](https://help.jasmin.ac.uk/article/4881-lotus-queues) by submitting it to [the SLURM scheduler](https://help.jasmin.ac.uk/article/4880-batch-scheduler-slurm-overview). For this purpose, the `extract.sbatch` job submission script is provided. Use it like this:

```
sbatch extract.sbatch
```

Note that inside this script, the region to process is hard-coded with arguments `--minlon`, `--maxlon`, `--minlat`, `--maxlat`, as is the name of the output file (the unnamed argument to `extract.py`). If you would want to operate on a different region, either modify the script in-place, or create a copy and modify that.

### Running the Community Size Spectrum Model




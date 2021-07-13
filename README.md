# NEMO-MEDUSA-CSSM

This repository contains scripts for simulation with the Community Size Spectrum Model, driven by the 1/4 degree "ROAM" projections from NEMO-MEDUSA.

## Running on JASMIN

### Set up

After logging into one of JASMIN's [scientific analysis servers (e.g., `sci2.jasmin.ac.uk`)](https://help.jasmin.ac.uk/article/121-sci-servers), first try running `conda`.
If you have previouly used [JASMIN's Python environment](https://help.jasmin.ac.uk/article/4729-jaspy-envs), this command should work.
Alternatively, if you get "command not found", do the following:

```
module load jaspy
conda init bash
```

Then log out and back in. Now `conda` should work. You will not need `module load jaspy` anymore, as `conda init bash` has added the necessary initialization logic to your `.bashrc`. In fact, if you would execute `module load jaspy` again, it breaks your conda environment (that may be a bug that ultimately gets addressed by the JASMIN/Jaspy team - it is an issue on 7 July 2021).

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

This sets up a new, isolated Python environment. Note that this can later be removed if needed with `conda env remove -n nemo-medusa-cssm`.

### Extracting NEMO-MEDUSA outputs

This is done by Python script `extract.py`. As this a non-parallelized job that can take over 24 hours, it is done on [the `long-serial` queue](https://help.jasmin.ac.uk/article/4881-lotus-queues) by submitting it to [the SLURM scheduler](https://help.jasmin.ac.uk/article/4880-batch-scheduler-slurm-overview). For this purpose, the `extract.sbatch` job submission script is provided. Use it like this:

```
sbatch extract.sbatch
```

**Notes:**
* Inside this script, the region to process is hard-coded with arguments `--minlon`, `--maxlon`, `--minlat`, `--maxlat`, as is the name of the output file (the unnamed argument to `extract.py`). If you want to extract data for a different region or change the name of the output file, either modify the script in-place, or create a copy and modify that.
* This script can take a long time! For instance, extracting 1993-2099 at monthly resolution for 20 - 78 degrees East, -38 - 25 degrees North takes 12-15 hours (and on a weekday in July it first spent 19 hours waiting in the queue!) The current maximum runtime is set to 48 hours in `extract.sbatch` with the line `#SBATCH --time=48:00:00`. You may want to change that, e.g., when extracting data at higher temporal resolution. That can be done by editing `extract.sbatch`, or by specifying a custom maximum runtime to sbatch with argument `--time=<HH:MM:SS>`. [Note that the `long-serial` queue currently has a maximum runtime of 168 hours = 7 days.](https://help.jasmin.ac.uk/article/4881-lotus-queues)

After submitting the job, you can check its status with

```
squeue -l -u $USER
```

The output of this command will show the job identifier ("JOBID").
The output of the job itself will be written to `<JOBID>.out` and `<JOBID>.err`.
You can keep an eye on the progress of the job with `tail -f <JOBID>.out`.

When the job completes, it should have created a single NetCDF file. The name of this file is set in `extract.sbatch` (the unnamed argument to `extract.py`). You can verify whether the job completed successfully by verifying whether the output file (`<JOBID>.out`) ends with "Extraction complete".

### Running the Community Size Spectrum Model

This is done by Python script `run.py`. Each horizontal grid point is processed independently, as there is no horizontal exchange or movement of predators between grid cells. The simulation is parallized using [Parallel Python](https://www.parallelpython.com/), which farms out each task (one per grid point) to nodes allocated by the queuing system. As soon as one task completes on a node/core, the next is started. Thus, the more nodes/cores you allocate to the job, the quicker it will be done - with a maximum equal to the total number of grid points.

On JASMIN, simulations are done on [the `par-multi` queue](https://help.jasmin.ac.uk/article/4881-lotus-queues) by submitting it to [the SLURM scheduler](https://help.jasmin.ac.uk/article/4880-batch-scheduler-slurm-overview). For this purpose, the `run.sbatch` job submission script is provided. Use it like this:

```
sbatch run.sbatch
```

**Notes:**
* Inside `run.sbatch`, the file to operate upon use is configured with the line `name=nemo-medusa-WIO`. This "name" variable is picked up in the final line that calls `run.py`; it sets the input file to `nemo-medusa-WIO.nc` (in the local directory), and the output file to `results/nemo-medusa-WIO.nc`. You can change the "name" variable, or if needed, the input and output paths separately, by editing (a copy of) `run.sbatch`.
* You can customize the number of nodes and the maximum runtime by editing `run.sbatch`, or by providing additional arguments `--nodes=<N>` and `--time=<HH:MM:SS>` to `sbatch`.

You can check the status of your job as described in the previous section.

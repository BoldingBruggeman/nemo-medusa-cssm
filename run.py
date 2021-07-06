from __future__ import annotations

import sys
import os
import glob
import datetime
import argparse
import re
import shutil
from typing import Optional

import yaml

import numpy
from matplotlib import pyplot
from matplotlib.dates import datestr2num, date2num, num2date
import netCDF4

start_time = None #datetime.datetime(2009, 1, 1)
stop_time = None #datetime.datetime(2012, 1, 1)

sys.path.insert(0, './extern/fabm-mizer/python')
import mizer

# Function for converting from Equivalent Spherical Diameter (micrometer) to wet mass in g
def esd2mass(*d): # d: equivalent spherical diameter in micrometer
    V = 4. / 3. * numpy.pi * (numpy.array(d) / 2e6)**3  # V: volume in m3
    return V * 1e6  # mass in g approximately equals volume in m3 multiplied by 1e6 (assumes density of 1000 kg/m3)

preylist = []
preylist.append(('diatoms', 'PHD', esd2mass(20., 200.), 6.625))
preylist.append(('non-diatoms', 'PHN',   esd2mass(2., 20.), 6.625))
preylist.append(('microzooplankton', 'ZMI', esd2mass(20., 200.), 5.625))
preylist.append(('mesozooplankton', 'ZME', (1e-5, 1e-3), 5.625))
temp_name = 'votemper'
time_name = 'time_counter'

# mizer parameters
parameters = {
    'w_min': 1e-3,
    'w_inf': 1e6,
    'nclass': 100,
    'T_dependence': 1,
    'T_ref': 13.,
    'E_a': 0.63,
    'beta': 100,
    'sigma': float(numpy.log(10.)),   # paper has log10 units, we use ln units
    'gamma': 156, # clearance in m3/yr for single individual of mass 1 g. Blanchard et al 2009: 640 m3/yr; Blanchard et al 2012: 64 ?UNITS? [times kappa=0.5 for time spent in pelagic]; Faking giants paper gives 10^14.076 * W^0.926 * exp(-Ea/(kT) L d-1, which is 428 L d-1 = 156 m3 yr-1
    'q': 0.82,
    'alpha': 0.2,
    'z0_type': 1,
    'z0pre': 0.1,
    'z0exp': -0.25,
    'w_s': 1000.,
    'z_s': 0.3,
    'ks': 0.,
    'SRR': 0,
    'recruitment': 0.,
    'h': 1e9,
    'fishing_type': 1,
    'w_minF': 1.25, # Blanchard et al 2012
    'F': 0.4 #8  # note: need to put double the intended value due to fisheries equation!
}

def add_variable(nc: netCDF4.Dataset, name: str, long_name: str, units: str, data=None, dimensions: Optional[tuple[str, ...]]=None, zlib: bool=False, contiguous: bool=True):
    if dimensions is None:
        dimensions = (time_name,)
    chunksizes = [1] * len(dimensions)
    if time_name in dimensions:
        chunksizes[dimensions.index(time_name)] = len(nc.dimensions[time_name])
    ncvar = nc.createVariable(name, float, dimensions, zlib=zlib, fill_value=-2e20, contiguous=contiguous, chunksizes=chunksizes)
    if data is not None:
        ncvar[:] = data
    ncvar.long_name = long_name
    ncvar.units = units
    if 'x' in dimensions and 'y' in dimensions and 'nav_lon' in nc.variables and 'nav_lat' in nc.variables:
       ncvar.coordinates = 'nav_lon nav_lat'
    return ncvar

def copy_variable(nc: netCDF4.Dataset, ncvar: netCDF4.Variable, **kwargs):
   ncvar_out = nc.createVariable(ncvar.name, ncvar.dtype, ncvar.dimensions, fill_value=getattr(ncvar, '_FillValue', None), **kwargs)
   for key in ncvar.ncattrs():
      if key != '_FillValue':
         setattr(ncvar_out, key, getattr(ncvar, key))
   if 'x' in ncvar.dimensions and 'y' in ncvar.dimensions and 'nav_lon' in nc.variables and 'nav_lat' in nc.variables:
      ncvar_out.coordinates = 'nav_lon nav_lat'
   ncvar_out[...] = ncvar[...]
   return ncvar_out

def process_location(args: tuple[str, int, int]):
    path, i, j = args
    print('Processing %s for i=%i, j=%i...' % (path, i, j))

    prey = []
    for name, ncname, size_range, CN_ratio in preylist:
        scale_factor = 10  * 0.001 * 12 * CN_ratio # 10 g wet mass/g carbon * 0.001 g C/mg C * 12 mg C/mmol C * mmol C/mmol N
        timeseries = mizer.datasources.TimeSeries(path, ncname, scale_factor=scale_factor, time_name=time_name, x=i, y=j, stop=stop_time)
        times = timeseries.times
        prey.append(mizer.Prey(name, size_range, timeseries))
    prey_collection = mizer.PreyCollection(*prey)
    prey_collection = mizer.GriddedPreyCollection(prey_collection)

    # environment: temperature and depth of the layer over which fish interact
    temp = mizer.datasources.TimeSeries(path, temp_name, time_name=time_name, x=i, y=j, stop=stop_time)
    depth = mizer.datasources.TimeSeries(path, 'bm_int**2/bm2_int', time_name=time_name, x=i, y=j, stop=stop_time)

    # create mizer model
    m = mizer.Mizer(prey=prey_collection, parameters=parameters, temperature=temp, recruitment_from_prey=True, depth=depth)

    # Time-integrate
    spinup = 50
    istart = 0 if start_time is None else times.searchsorted(date2num(start_time))
    istop = len(times) if stop_time is None else times.searchsorted(date2num(stop_time))
    times = times[istart:istop]

    result = m.run(times, spinup=spinup, verbose=True, save_spinup=False, save_loss_rates=True)

    if result is None:
        return

    biomass = result.get_biomass_timeseries()
    landings_var, landings = result.get_timeseries('landings')
    lfi1 = result.get_lfi_timeseries(1.)
    lfi80 = result.get_lfi_timeseries(80.)
    lfi250 = result.get_lfi_timeseries(250.)
    lfi500 = result.get_lfi_timeseries(500.)
    lfi10000 = result.get_lfi_timeseries(10000.)
    landings[1:] = landings[1:] - landings[:-1]
    landings[0] = 0
    return path, i, j, times, biomass, landings, lfi1, lfi80, lfi250, lfi500, lfi10000, m.bin_masses, result.spectrum, result.get_loss_rates()

def parallel_process_location(args, p):
    import run
    run.parameters = p
    return run.process_location(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path')
    parser.add_argument('output_path')
    parser.add_argument('--ncpus', type=int, default=None)
    parser.add_argument('--ppservers', default=None)
    parser.add_argument('--secret', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parameters', default=None)
    parser.add_argument('--shm', action='store_true')
    parser.add_argument('--ntask', type=int)
    args = parser.parse_args()

    if args.parameters is not None:
        with open(args.parameters, 'rU') as f:
            args.parameters = yaml.safe_load(f)
        parameters = args.parameters

    if isinstance(args.ppservers, (str, u''.__class__)):
        match = re.match(r'(.*)\[(.*)\](.*)', args.ppservers)
        if match is not None:
            # Hostnames in PBS/SLURM notation, e.g., node[01-06]
            ppservers = []
            left, middle, right = match.groups()
            for item in middle.split(','):
                if '-' in item:
                    start, stop = item.split('-')
                    for i in range(int(start), int(stop)+1):
                        ppservers.append('%s%s%s' % (left, str(i).zfill(len(start)), right))
                else:
                    ppservers.append('%s%s%s' % (left, item, right))
        else:
            # Comma-separated hostnames
            ppservers = args.ppservers.split(',')
        ppservers = tuple(ppservers)
    else:
        assert args.ppservers is None
        ppservers = ()
    if args.ncpus is None:
        args.ncpus = 'autodetect'

    tasks = []
    if not os.path.isdir(args.output_path):
       os.mkdir(args.output_path)
    for path in glob.glob(args.source_path):
        with netCDF4.Dataset(path) as nc:
            if 'mask' in nc.variables:
                mask = nc.variables['mask'][...] > 0
            else:
                mask = (nc.variables[temp_name][...] > 0).any(axis=0)
            for i in range(len(nc.dimensions['x'])):
                for j in range(len(nc.dimensions['y'])):
                    if mask[j, i]:
                        tasks.append((path, i, j))
    if args.ntask is not None:
        tasks = tasks[:args.ntask]

    source2output = {}
    def get_output(source: str, times, bin_masses: numpy.ndarray, compress: bool=False, add_biomass_per_bin: bool=False, contiguous: bool=False, save_loss_rates: bool=False):
        if source not in source2output:
            with netCDF4.Dataset(path) as nc:
                output_path = os.path.join(args.output_path, os.path.basename(source))
                print('Creating %s....' % output_path)
                ncout = netCDF4.Dataset(output_path, 'w')
                nctime_in = nc.variables[time_name]
                ncout.createDimension(time_name, len(times))
                ncout.createDimension('x', len(nc.dimensions['x']))
                ncout.createDimension('y', len(nc.dimensions['y']))
                ncout.createDimension('bin', len(bin_masses))
                nctime_out = ncout.createVariable(time_name, nctime_in.datatype, nctime_in.dimensions, zlib=compress, contiguous=contiguous)
                nctime_out.units = nctime_in.units
                dates = [dt.replace(tzinfo=None) for dt in num2date(times)]
                nctime_out[...] = netCDF4.date2num(dates, nctime_out.units)
                if 'nav_lon' in nc.variables:
                    copy_variable(ncout, nc.variables['nav_lon'], zlib=compress)
                    copy_variable(ncout, nc.variables['nav_lat'], zlib=compress)
                add_variable(ncout, 'biomass', 'biomass', 'g WM/m2', dimensions=(time_name, 'y', 'x'), zlib=compress, contiguous=contiguous)
                add_variable(ncout, 'landings', 'landings', 'g WM', dimensions=(time_name, 'y', 'x'), zlib=compress, contiguous=contiguous)
                ncwm = add_variable(ncout, 'binmass', 'wet mass per individual', 'g WM', dimensions=('bin',), zlib=compress, contiguous=contiguous)
                ncwm[:] = bin_masses
                for wm in (1, 80, 250, 500, 10000):
                    add_variable(ncout, 'lfi%i' % wm, 'fraction of fish > %i g' % wm, '-', dimensions=(time_name, 'y', 'x'), zlib=compress, contiguous=contiguous)
                if add_biomass_per_bin:
                    add_variable(ncout, 'Nw', 'biomass per bin', 'g WM/m2', dimensions=(time_name, 'y', 'x', 'bin'), zlib=compress, contiguous=contiguous)
                if save_loss_rates:
                    add_variable(ncout, 'loss', 'loss rate per bin', 'd-1', dimensions=(time_name, 'y', 'x', 'bin'), zlib=compress, contiguous=contiguous)
            source2output[source] = ncout
        return source2output[source]

    def save_result(result, sync: bool=True, add_biomass_per_bin: bool=False, save_loss_rates: bool=False):
        source, i, j, times, biomass, landings, lfi1, lfi80, lfi250, lfi500, lfi10000, bin_masses, spectrum, loss_rates = result
        assert spectrum.shape == loss_rates.shape
        print('saving results from %s, i=%i, j=%i' % (source, i, j))
        ncout = get_output(source, times, bin_masses, add_biomass_per_bin=add_biomass_per_bin, save_loss_rates=save_loss_rates)
        ncout.variables['biomass'][:, j, i] = biomass
        ncout.variables['landings'][:, j, i] = landings
        ncout.variables['lfi1'][:, j, i] = lfi1
        ncout.variables['lfi80'][:, j, i] = lfi80
        ncout.variables['lfi250'][:, j, i] = lfi250
        ncout.variables['lfi500'][:, j, i] = lfi500
        ncout.variables['lfi10000'][:, j, i] = lfi10000
        if add_biomass_per_bin:
            ncout.variables['Nw'][:, j, i, :] = spectrum[:, :]
        if save_loss_rates:
            ncout.variables['loss'][:, j, i, :] = loss_rates[:, :]
        if sync:
           ncout.sync()

    job_server = None
    final_output_path = None
    if args.ncpus == 1:
        import cProfile
        import pstats
        def runSerial(n):
            for i in range(n):
                save_result(parallel_process_location(tasks[i], parameters), add_biomass_per_bin=True, save_loss_rates=True)
        cProfile.run('runSerial(%s)' % min(len(tasks), 3), 'mizerprof')
        p = pstats.Stats('mizerprof')
        p.strip_dirs().sort_stats('cumulative').print_stats()
    else:
        if args.debug:
            import logging
            logging.basicConfig( level=logging.DEBUG)
        import pp
        if args.shm:
            final_output_path = args.output_path
            args.output_path = '/dev/shm'
        job_server = pp.Server(ncpus=args.ncpus, ppservers=ppservers, restart=True, secret=args.secret)
        jobs = []
        for task in tasks:
            jobs.append(job_server.submit(parallel_process_location, (task, parameters)))
        ijob = 0
        while jobs:
            job = jobs.pop(0)
            result = job()
            if result is not None:
               print('job %i: saving result...' % ijob)
               save_result(result, sync=ijob % 1000 == 0, add_biomass_per_bin=True, save_loss_rates=True)
            else:
               print('job %i: FAILED!' % ijob)
            ijob += 1
        job_server.print_stats()

    for source, nc in source2output.items():
        name = os.path.basename(source)
        print('Closing %s...' % os.path.join(args.output_path, name))
        nc.close()
        if final_output_path is not None:
           target = os.path.join(final_output_path, name)
           if os.path.isfile(target):
              os.remove(target)
           shutil.move(os.path.join(args.output_path, name), target)

    if job_server is not None:
       job_server.destroy()
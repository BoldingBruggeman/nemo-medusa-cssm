from __future__ import annotations

import argparse
import glob
import os.path
import sys
import numpy
import netCDF4
from typing import Optional, Union
from collections.abc import Iterable

medusa_names = 'PHN', 'PHD', 'ZMI', 'ZME'
temp_name = 'votemper'
thickness = numpy.array("""1.0200 	    1.0800	    1.1500	    1.2300	    1.3400	    1.4700	    1.6300	    1.8300	    2.0800	    2.3700	    2.7100	    3.1100	    3.5600	    4.0500	    4.5900	    5.1500	    5.7300	    6.3300	    6.9500	    7.5800	    8.2400	    8.9400	    9.7000	   10.5300	   11.4600	   12.5000	   13.6800	   15.0100	   16.5400	   18.2700	   20.2500	   22.5000	   25.0500	   27.9400	   31.1900	   34.8300	   38.8900	   43.3900	   48.3500	   53.7600	   59.6200	   65.9200	   72.6100	   79.6600	   87.0000	   94.5600	  102.2600	  110.0100	  117.7100	  125.2900	  132.6400	  139.7100	  146.4300	  152.7500	  158.6400	  164.0800	  169.0600	  173.5800	  177.6700	  181.3300	  184.6000	  187.5000	  190.0600	  192.3100	  194.2900	  196.0200	  197.5300	  198.8400	  199.9800	  200.9700	  201.8300	  202.5700	  203.2000	  203.7500	  204.2300""".split('\t'), dtype=float)

def check_file(path: str, required: Iterable) -> bool:
   with netCDF4.Dataset(path) as nc:
      print('%s... ' % path, end='')
      missing = [variable for variable in required if variable not in nc.variables]
      print('OK' if not missing else 'MISSING %s' % (' ,'.join(missing)))
      return not missing

compress = False
contiguous = False
chunk = True

def find_time_variable(nc: netCDF4.Dataset) -> netCDF4.Variable:
   for ncvar in nc.variables.values():
      if getattr(ncvar, 'standard_name', None) == 'time':
         return ncvar
   raise Exception('Time coordinate not found.')

def get_time(nc: str) -> tuple[str, str, str, numpy.ndarray]:
   with netCDF4.Dataset(nc) as nc:
      ncvar = find_time_variable(nc)
      return ncvar.name, ncvar.units, ncvar.calendar, netCDF4.num2date(ncvar[:], ncvar.units, ncvar.calendar)

def copy_variable(ncout: netCDF4.Variable, ncvar: netCDF4.Variable, dimensions: Optional[tuple[str, ...]]=None, **kwargs_in):
   if dimensions is None:
      dimensions = ncvar.dimensions
   kwargs = {}
   if chunk and 'time_counter' in dimensions:
      kwargs['chunksizes'] = [{'time_counter': ntime}.get(dim, 1) for dim in dimensions]
   if hasattr(ncvar, '_FillValue'):
      kwargs['fill_value'] = ncvar._FillValue
   for key, value in kwargs_in.items():
      if key != 'name':
         kwargs[key] = value
   ncvar_out = ncout.createVariable(kwargs_in.get('name', ncvar.name), ncvar.dtype, dimensions, zlib=compress, contiguous=contiguous, **kwargs)
   for key in ncvar.ncattrs():
      if key != '_FillValue':
         setattr(ncvar_out, key, getattr(ncvar, key))
   if 'x' in dimensions and 'y' in dimensions:
      ncvar_out.coordinates = 'nav_lon nav_lat'
   return ncvar_out

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('target')
   parser.add_argument('--source', default='/gws/nopw/j04/nemo_vol2/ROAM/????/*_1m_*_ptrc_T_*.nc')
   parser.add_argument('--minlat', type=float, default=-91)
   parser.add_argument('--maxlat', type=float, default=91)
   parser.add_argument('--minlon', type=float, default=-361)
   parser.add_argument('--maxlon', type=float, default=361)
   parser.add_argument('--check', action='store_true')
   parser.add_argument('--resume', action='store_true')
   parser.add_argument('-n', type=int, help='Number of files to process (for debugging only)')
   arguments = parser.parse_args()

   paths = glob.glob(arguments.source)
   print('%i files found.' % len(paths))
   if not paths:
      sys.exit(1)
   paths.sort()
   if arguments.n is not None:
      paths = paths[:arguments.n]

   valid = True
   if arguments.check:
      for path in paths:
         valid = check_file(path, medusa_names) and valid
         valid = check_file(path.replace('_ptrc_T_', '_grid_T_'), (temp_name,)) and valid
   if not valid:
      sys.exit(1)

   print('Discovering time period:')
   time_name, time_units, time_calendar, last_time = None, None, None, None
   start, stop = None, None
   ntime = 0
   for path in paths:
      print('  - %s' % path)
      current_time_name, current_time_units, current_time_calendar, current_time = get_time(path)
      if time_name is None:
         time_name, time_units, time_calendar = current_time_name, current_time_units, current_time_calendar
      if current_time_name != time_name:
         print('    WARNING: different time coordinate name %s (first file(s) use %s). Value(s): %s' % (current_time_name, time_name, current_time))
      if last_time is not None:
         assert current_time[0] > last_time
      assert time_units == current_time_units
      assert time_calendar == current_time_calendar
      start = current_time[0] if start is None else min(start, current_time[0])
      stop = current_time[-1] if stop is None else max(stop, current_time[-1])
      ntime += current_time.size
      last_time = current_time[-1]
   print('Time range: %s - %s' % (start.strftime('%Y-%m-%d'), stop.strftime('%Y-%m-%d')))
   print('Time count: %i' % ntime)

   #sys.exit(0)

   def read_cyclic(ncvar, imin: int=0, imax: Optional[int]=None, preslice=(Ellipsis,)):
      shift = ncvar.shape[-1] // 2
      if imax is None:
         imax = ncvar.shape[-1]
      real_start = imin + shift
      real_stop = imax + shift
      left_start = max(0, real_start - ncvar.shape[-1])
      left_stop = max(0, real_stop - ncvar.shape[-1])
      left = ncvar[preslice + (slice(left_start, left_stop),)]
      right_start = min(real_start, ncvar.shape[-1])
      right_stop = min(real_stop, ncvar.shape[-1])
      right = ncvar[preslice + (slice(right_start, right_stop),)]
      return numpy.concatenate((right, left), axis=-1)

   masked_lonlat_value = -9999.
   grid_file = os.path.abspath(os.path.join(os.path.dirname(arguments.source), '..', '..', 'domain/mesh_zgr.nc'))
   masked_grid = False
   if not os.path.isfile(grid_file):
      print('WARNING: grid file %s not found. Reading coordinates from %s.' % (grid_file, paths[0]))
      grid_file = paths[0]
      masked_grid = True
   with netCDF4.Dataset(grid_file) as nc:
      nc.set_auto_mask(False)
      lon = read_cyclic(nc.variables['nav_lon'])
      lat = read_cyclic(nc.variables['nav_lat'])
   if masked_grid:
      mask = numpy.logical_and(lon == 0., lat == 0.)
      lon[mask] = masked_lonlat_value
      lat[mask] = masked_lonlat_value
   lat_ma = numpy.ma.masked_equal(lat, masked_lonlat_value)
   lon_ma = numpy.ma.masked_equal(lon, masked_lonlat_value)
   ny, nx = lon.shape
   print('Input grid:')
   print('  size: nx=%i x ny=%i' % (nx, ny))
   print('  longitude: %s - %s' % (lon_ma.min(), lon_ma.max()))
   print('  latitude: %s - %s' % (lat_ma.min(), lat_ma.max()))
   print('Requested region:')
   print('  longitude: %s - %s' % (arguments.minlon, arguments.maxlon))
   print('  latitude: %s - %s' % (arguments.minlat, arguments.maxlat))

   valid_lon = numpy.logical_and(lon >= arguments.minlon, lon <= arguments.maxlon)
   valid_lat = numpy.logical_and(lat >= arguments.minlat, lat <= arguments.maxlat)
   valid = numpy.logical_and(valid_lon, valid_lat)
   iy, ix = valid.nonzero()

   imin, imax = ix.min(), ix.max() + 1
   jmin, jmax = iy.min(), iy.max() + 1
   print('Slicing:')
   print('  x: %i - %i' % (imin, imax))
   print('  y: %i - %i' % (jmin, jmax))
   lat = lat[jmin:jmax, imin:imax]
   lon = lon[jmin:jmax, imin:imax]
   lat_ma = numpy.ma.masked_equal(lat, masked_lonlat_value)
   lon_ma = numpy.ma.masked_equal(lon, masked_lonlat_value)
   ny, nx = lon.shape
   print('Output grid:')
   print('  size: nx=%i x ny=%i' % (nx, ny))
   print('  longitude: %s - %s' % (lon_ma.min(), lon_ma.max()))
   print('  latitude: %s - %s' % (lat_ma.min(), lat_ma.max()))

   mode = 'r+' if os.path.isfile(arguments.target) and arguments.resume else 'w'
   iout = 0
   ncvariables_out = None
   with netCDF4.Dataset(arguments.target, mode, clobber=False, diskless=True, persist=True) as ncout:
      istart = 0
      if mode != 'w':
         nctime_out = ncout.variables['time_counter']
         ncw_int_out = ncout.variables['bm_int']
         ncw2_int_out = ncout.variables['bm2_int']
         istart = nctime_out.size - 2
         ncvariables_out = [ncout.variables[variable] for variable in (temp_name,) + medusa_names]
      for path in paths:
         print('Reading %s...' % path)
         physics_path = path.replace('_ptrc_T_', '_grid_T_')

         with netCDF4.Dataset(path) as nc, netCDF4.Dataset(physics_path) as nc_physics:
            nctime = find_time_variable(nc)
            times = netCDF4.num2date(nctime[:], nctime.units, nctime.calendar)
            nctemp = nc_physics.variables[temp_name]

            # Find target variables (prey biomasses)
            ncvariables = [nctemp]
            for variable in medusa_names:
               assert variable in nc.variables, 'Variable %s not found in %s. Available: %s' % (variable, path, ', '.join(nc.variables.keys()))
               ncvariables.append(nc.variables[variable])

            first = ncvariables_out is None
            if first:
               # We are creating the output file. Define dimensions, variables, copy coorinates, etc.
               ncout.createDimension('time_counter', ntime)
               ncout.createDimension('x', nx)
               ncout.createDimension('y', ny)
               copy_variable(ncout, nc.variables['nav_lat'], fill_value=masked_lonlat_value)[:, :] = lat_ma
               copy_variable(ncout, nc.variables['nav_lon'], fill_value=masked_lonlat_value)[:, :] = lon_ma
               nctime_out = copy_variable(ncout, nctime, name='time_counter')
               kwargs = {}
               if chunk:
                  kwargs['chunksizes'] = (ntime, 1, 1)
               ncw_int_out = ncout.createVariable('bm_int', 'f', ('time_counter', 'y', 'x'), zlib=compress, contiguous=contiguous, fill_value=1e20, **kwargs)
               ncw2_int_out = ncout.createVariable('bm2_int', 'f', ('time_counter', 'y', 'x'), zlib=compress, contiguous=contiguous, fill_value=1e20, **kwargs)
               ncw_int_out.coordinates = 'nav_lon nav_lat'
               ncw2_int_out.coordinates = 'nav_lon nav_lat'
               ncw_int_out.units = '(%s) m' % ncvariables[0].units
               ncw2_int_out.units = '(%s)2 m' % ncvariables[0].units
               ncvariables_out = [copy_variable(ncout, ncvar, dimensions=('time_counter', 'y', 'x')) for ncvar in ncvariables]

            for itime, time in enumerate(times):
               print('  Processing %s...' % time.strftime('%Y-%m-%d'))
               if iout >= istart:
                  alldata = [read_cyclic(ncvariable, imin, imax, preslice=(itime, Ellipsis, slice(jmin, jmax))) for ncvariable in ncvariables]
                  biomass = sum(alldata[1:]) # skip temperature!
                  weights = biomass * thickness[:, numpy.newaxis, numpy.newaxis]
                  weights_int = weights.sum(axis=0)
                  weights2_int = (biomass * weights).sum(axis=0)
                  ncw_int_out[iout, :, :] = weights_int
                  ncw2_int_out[iout, :, :] = weights2_int
                  weights /= weights_int[numpy.newaxis, :, :]
                  for ncvariable_out, data in zip(ncvariables_out, alldata):
                     ncvariable_out[iout, :, :] = (weights * data).sum(axis=0)
                  nctime_out[iout] = nctime[itime]
               iout += 1
            if iout >= istart:
               ncout.sync()

   print('Extraction complete')

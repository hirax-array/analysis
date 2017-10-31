'''
This script makes waterfall style plots of the visibilites for all baselines in all data files in a specified directory.

It is designed to be run by a cron job to create up to date data quality plots.

A log of plotted files is saveed for each acq directory, so that they can be skipped the next time the script is run.

All plots are saved in subdirectories for their baseline, for ease of monitoring a baseline over time.

The  currently only runs for a single acq directory (Files produced by a single instance of kotekan.) If kotekan is restarted, a new directory must be specified. This could easily be generalized to all directories in the base data location.

Created: 2017-09-26 Ben Saliwanchik
Modified: 2017-10-03 
'''

import os, glob, pickle, time, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates
from astropy.time import Time
import multiprocessing
from functools import partial


########################################
def data_quality():
  tic = time.time()

  #Currently runs for one acq directory, can generalize to whole base dir
  #data_base_dir = '/data/hiraxacq/hirax_data/untransposed'
  data_base_dir = '/home/bens/hirax_data_tmp'

  #data_dir = data_base_dir+'/20170921T170320Z_pathfinder_corr'
  data_dir = data_base_dir+'/20171012T163621Z_hirax8_corr'
  #data_dir = '/home/bens/hirax_data_tmp' #For testing

  data_quality_dir = data_dir+'/data_quality'
  if not os.path.exists(data_quality_dir):
    os.mkdir(data_quality_dir)

  data_files_tmp = glob.glob(data_dir+'/*.h5')

  #Check logs to see which files have already been plotted
  logfile = data_quality_dir+'/files_plotted.pkl'
  if os.path.isfile(logfile):
    f = open(logfile, 'rb')
    files_plotted = pickle.load(f)
    f.close()
  else:
    files_plotted = []
  data_files = filter(lambda x: x not in files_plotted, data_files_tmp)
  if data_files: 
    print 'Unplotted data files found.'
  else:
    print 'No unplotted data files found. Exiting.'

  #Make plots for all unplotted data files in directory
  for data_file_name in data_files:
    print 'Opening data file: '+data_file_name
    data_file = h5py.File(data_file_name, 'r')

    #Baseline number strings for plot labels
    prod = zip(*data_file['index_map']['prod'])
    prod_0 = map(str, prod[0])
    prod_1 = map(str, prod[1])
    baseline_nums = map(lambda x,y:x+'-'+y, prod_0,prod_1)
    n_baselines = len(baseline_nums)

    band_centers = data_file['index_map']['freq']['centre']
    unix_times = data_file['index_map']['time']['ctime']

    #Will plot in parallel
    #Faster to only pass one baseline to each parallel plotting instance
    #print 'Loading baseline data.'
    #baseline_data = [data_file['vis'][:,:,n] for n in range(0,n_baselines)]
  
    #Bundle all baseline data together as single iterable for pool
    print 'Bundling baseline data for parallel processing.'
    baselines = [[baseline_nums[n], data_file['vis'][:,:,n]] for n in range(0,n_baselines)]

    #Close file once data is loaded 
    data_file.close()


    #Make plots
    #Pool() defaults to total number of cores
    print 'Plotting! Baselines plotted: '
    pool = multiprocessing.Pool()
    partial_plot_vis = partial(plot_vis, data_quality_dir, band_centers, unix_times) 
    pool.map(partial_plot_vis, baselines)
    pool.close()
    pool.join()

    #Add data file name to list of plotted files if it is a complete observation 
    #(Want to replot partially complete obs transferred by rsync)
    if len(unix_times == 1024):  
      files_plotted.append(data_file_name)

    #Save list of plotted files
    f = open(logfile, 'wb')
    pickle.dump(files_plotted, f)
    f.close()
    print 'Done with file: '+data_file_name

  time_elapsed = time.time()-tic
  print 'Time elapsed: '+str(round(time_elapsed,3))+'s'

########################################




########################################
#Plot visibilities for a single baseline
def plot_vis(data_quality_dir, band_centers, unix_times, baseline):
    #Set up data for plot
    baseline_num = baseline[0]
    print baseline_num
    datetimes = Time(unix_times, format='unix').datetime
    iso_times = Time(unix_times, format='unix').iso
    isot_times = Time(unix_times, format='unix').isot
    to_plot = baseline[1]['r']-np.median(baseline[1]['r'])  

    #Specify plot details
    fig, axis = plt.subplots()
    axis.set_title('Visiblities for Baseline '+baseline_num)
    extent = (band_centers[0], band_centers[-1], dates.date2num(datetimes[-1]), dates.date2num(datetimes[0]))
    aspect = np.abs((extent[1] - extent[0]) / (extent[3] - extent[2]))
    vmin, vmax = np.percentile(to_plot, (15, 85))
    _ = axis.imshow(to_plot, vmin=vmin, vmax=vmax, cmap='RdBu_r', aspect=aspect, extent=extent)
    axis.yaxis_date()
    axis.set_ylabel('UTC')
    axis.yaxis.set_major_locator(dates.MinuteLocator(byminute=[0, 30]))
    axis.yaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    # Annotate with date of observation start
    axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
    axis.set_xlabel('Frequency [GHz]')

    #Save and close figure
    baseline_dir = data_quality_dir+'/baseline_'+baseline_num
    if not os.path.exists(baseline_dir):
      os.mkdir(baseline_dir)
    plt.savefig(baseline_dir+'/vis_'+baseline_num+'_'+isot_times[0]+'.png')    
    plt.close()

########################################




if __name__ == '__main__':
  data_quality()

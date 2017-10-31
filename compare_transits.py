'''
This script creates plots comparing transit properties over a period of time.

Created: 2017-10-27 Ben Saliwanchik
Modified: 2017-10-27
'''

import os, glob, pickle, time, h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy import exp,asarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates
from astropy.time import Time
import datetime, logging

import multiprocessing
from functools import partial

#Run fitting and plotting in parallel
#Pool() defaults to total number of cores
#pool = multiprocessing.Pool()
#partial_plot_transit = partial(plot_transit, data_quality_dir, band_centers, unix_times) 
#pool.map(partial_plot_transit, baselines)
#pool.close()
#pool.join()

data_dir = '/home/bens/hirax_analysis'
data_quality_dir = data_dir+'/data_quality'

#Open new log file for this run.
timestamp=str(datetime.datetime.now().isoformat().split('.')[0])
log_name = data_quality_dir+'/compare_transits_'+timestamp+'.log'
logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
  format="%(asctime)-15s %(levelname)-8s %(message)s")
# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

########################################
def compare_transits():

  data_files = glob.glob(data_quality_dir+'/fit_stats_*.pkl')
  if data_files == []:
    logging.warning('No data files found in '+data_quality_dir+'.')
    logging.warning('Exiting!')
    return
  iso_times, band_centers, baselines, x0, x0_uncert, sigma, sigma_uncert, fwhm, fwhm_uncert = ([] for i in range(9))
  for data_file_name in data_files:
    logging.info('Loading data file: '+data_file_name)
    f = open(data_file_name, 'rb')
    data_file = pickle.load(f)
    f.close()
    iso_times.append(data_file[0])
    band_centers.append(data_file[1])
    baselines.append(data_file[2])
    x0.append(data_file[3])
    x0_uncert.append(data_file[4])
    sigma.append(data_file[5])
    sigma_uncert.append(data_file[6])
    fwhm.append(2.0*np.sqrt(2.0*np.log(2.0))*np.array(data_file[5]))
    fwhm_uncert.append(2.0*np.sqrt(2.0*np.log(2.0))*np.array(data_file[6]))
  
  isot_times = Time(iso_times[0][0], format='iso').isot

  if not os.path.exists(data_quality_dir):
    os.mkdir(data_quality_dir)
    logging.info('Making data quality directory...')

  #Drift scan speed is: 0.00417807457 degrees/sec 
  #assuming a mean sidereal day of 23.9344699 hrs
  #Time b/w samples: 10.74+/-0.38s
  #Angular size of pixels/samples: 0.044862 degrees
  angles = np.array(range(len(baselines[0])))*0.044862
  times = np.array(range(len(baselines[0])))*10.74

  #Plot baseline comparison
  fig, axis = plt.subplots()
  axis.set_title('Baseline 6-6 Comparison')
  plot_colors = ['k', 'r', 'g', 'b', 'c']
  plot_offset = 0.1*max(baselines[0])
  for i in range(len(plot_colors)):
     _ = axis.plot(band_centers[i],baselines[i]+(i*plot_offset),plot_colors[i])
  axis.set_ylabel('Amplitude')
  axis.set_xlabel('Frequency [MHz]')

  #Save and close figure
  plt.savefig(data_quality_dir+'/baseline_6-6_comparison_'+isot_times+'.png')    
  plt.close()
  
  baseline_names = ['0-0', '1-1', '2-2', '3-3', '4-4', '5-5', '6-6', '7-7', '8-8', '9-9', '10-10', '11-11', '12-12', '13-13', '14-14', '15-15']
  n_times = range(len(x0))
  for  baseline_num in range(len(baseline_names)):
    #Plot x0 comparison
    x0_tmp = [x0[i][baseline_num] for i in n_times]
    x0_uncert_tmp = [x0_uncert[i][baseline_num] for i in n_times]
    fig, axis = plt.subplots()
    axis.set_title('Transit Center Comparison, Baseline '+baseline_names[baseline_num])
    _ = axis.errorbar(n_times,x0_tmp,yerr=x0_uncert_tmp,fmt='ok')
    axis.set_ylabel('Transit peak time')
    axis.set_xlabel('Days after '+str(iso_times[0][0]))

    #Save and close figure
    plt.savefig(data_quality_dir+'/transit_x0_comparison_'+baseline_names[baseline_num]+'_'+isot_times+'.png')    
    plt.close()

    #Plot sigma comparison
    sigma_tmp = [sigma[i][baseline_num] for i in n_times]
    sigma_uncert_tmp = [sigma_uncert[i][baseline_num] for i in n_times]
    fig, axis = plt.subplots()
    axis.set_title('Transit Width (Sigma) Comparison, Baseline '+baseline_names[baseline_num])
    _ = axis.errorbar(n_times,sigma_tmp,yerr=sigma_uncert_tmp,fmt='ok')
    axis.set_ylabel('Transit width /sigma [degrees]')
    axis.set_xlabel('Days after '+str(iso_times[0][0]))

  
    #Save and close figure
    plt.savefig(data_quality_dir+'/transit_sigma_comparison_'+baseline_names[baseline_num]+'_'+isot_times+'.png')    
    plt.close()

    #Plot FWHM comparison
    fwhm_tmp = [fwhm[i][baseline_num] for i in n_times]
    fwhm_uncert_tmp = [fwhm_uncert[i][baseline_num] for i in n_times]
    fig, axis = plt.subplots()
    axis.set_title('Transit Width (FWHM) Comparison, Baseline '+baseline_names[baseline_num])
    _ = axis.errorbar(n_times,fwhm_tmp,yerr=fwhm_uncert_tmp,fmt='ok')
    axis.set_ylabel('Transit FWHM [degrees]')
    axis.set_xlabel('Days after '+str(iso_times[0][0]))

    #Save and close figure
    plt.savefig(data_quality_dir+'/transit_fwhm_comparison_'+baseline_names[baseline_num]+'_'+isot_times+'.png')    
    plt.close()

########################################


if __name__ == '__main__':
  compare_transits()

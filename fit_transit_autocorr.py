'''
This script fits a Gaussian profile to a transit in autocorrelation data.

This is a work in progress!

To add:
*Currently, specify file. In future, search all autocorrelation files in directory and search for transits?
*Fit Gaussian+median+slope
*Remove RFI bands with standard deviation from median flag.
*Implement parallel processing?

Created: 2017-10-18 Ben Saliwanchik
Modified: 2017-10-31
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
data_file_name = data_dir+'/00000000_0000.h5' #Pic A
#data_file_name = data_dir+'/00088140_0000.h5' #Pic A
#data_file_name = data_dir+'/00176102_0000.h5' #Pic A
#data_file_name = data_dir+'/00253068_0000.h5' #Pic A
#data_file_name = data_dir+'/00341029_0000.h5' #Pic A
#data_file_name = data_dir+'/00033165_0000.h5' #Fornax A
data_quality_dir = data_dir+'/data_quality'

if not os.path.exists(data_quality_dir):
  os.mkdir(data_quality_dir)
  logging.info('Making data quality directory...')

#Open new log file for this run.
timestamp=str(datetime.datetime.now().isoformat().split('.')[0])
log_name = data_quality_dir+'/fit_transit_'+timestamp+'.log'
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
def fit_transit_autocorr():

  #Make plots for all unplotted data files in directory
  #for data_file_name in data_files:
  logging.info('Opening data file: '+data_file_name)
  data_file = h5py.File(data_file_name, 'r')

  #RMS cutoff 
  #cutoff = 3.5e6  #Old absolute value cutoff
  cutoff = 3  #In sigma

  #Baseline number strings for plot labels
  prod = zip(*data_file['index_map']['prod'])
  prod_0 = map(str, prod[0])
  prod_1 = map(str, prod[1])
  baseline_nums = map(lambda x,y:x+'-'+y, prod_0,prod_1)
  n_baselines = len(baseline_nums)
 
  band_centers = data_file['index_map']['freq']['centre']
  unix_times = data_file['index_map']['time']['ctime']
  iso_times = Time(unix_times, format='unix').iso
  isot_times = Time(unix_times, format='unix').isot
  logging.info('Data file from: '+iso_times[0])
  logging.info('  to: '+iso_times[-1])

  #Drift scan speed is: 0.00417807457 degrees/sec 
  #assuming a mean sidereal day of 23.9344699 hrs
  #Time b/w samples: 10.74+/-0.38s
  #Angular size of pixels/samples: 0.044862 degrees
  angles = np.array(range(len(unix_times)))*0.044862
  times = np.array(range(len(unix_times)))*10.74

  #baseline (14,14) is 133, baseline (15,15) is 135
  #baseline (12,12) is 126, baseline (13,13) is 130
  #baseline (10,10) is 115, baseline (11,11) is 121
  #baseline (8,8) is 100, baseline (9,9) is 108
  #baseline (6,6) is 81, baseline (7,7) is 91
  #baseline (5,5) is 70, baseline (4,4) is 58
  #baseline (2,2) is 31, baseline (3,3) is 45
  #baseline (0,0) is 0, baseline (1, 1) is 16

  autocorrelations = [0, 16, 31, 45, 58, 70, 81, 91, 100, 108, 115, 121, 126, 130, 133, 135]
  x0_med, x0_uncert, sigma_med, sigma_uncert = ([] for n in range(4))
  for baseline_to_run in autocorrelations:
    freq = 512
    baseline_num = baseline_nums[baseline_to_run]
    band_center = str(band_centers[freq])
    logging.info('Running baseline: '+baseline_num)
    logging.info('Test frequency: '+band_center+'MHz')  

    logging.info('Bundling baseline data for parallel processing...')
    baseline = data_file['vis'][:,freq,baseline_to_run]['r']
    baselines = data_file['vis'][:,:,baseline_to_run]['r']

    logging.info('Generating parameter estimates...')
    #Check if frequency bin is railed
    if np.median(baseline)!=max(baseline):
      y = baseline-np.median(baseline)
      x = angles
      x0_est =  np.where(y==max(y))[0][0]
      sigma_est = x0_est-np.where(y>0.68*np.array(max(y)))[0][0]
 
      logging.info('Fitting Gaussian profile at test frequency...')
      try:
        popt,pcov = curve_fit(gaussian,x,y,p0=[max(y),x0_est,sigma_est])
        logging.info('Gaussian fit values: ')
        logging.info(popt)

        logging.info('Plotting at test frequency...')
        plot_transit(data_quality_dir, unix_times, baseline_num, band_center, popt, x, y)
      except:
        logging.info('Fit failed at test frequency. Skipping plots.') 

    else:
      logging.info('Test frequency bin railed. Skipping plots.')

    #Now try at every frequency, and plot x0 and sigma as func of freq
    x0_list = []
    sigma_list = []
    rms_list = []
    n_railed = 0
    n_unfittable = 0
    logging.info('Fitting Gaussian profile at each frequency...')
    for freq in range(len(band_centers)):
      if np.median(baselines[:,freq])==max(baselines[:,freq]):
        n_railed += 1
        logging.debug(str(band_centers[freq])+'MHz bin railed. Skipping.')
        x0_list.append(0)
        sigma_list.append(0)
        rms_list.append(0)
        continue
      y = baselines[:,freq]-np.median(baselines[:,freq])
      x0_est =  np.where(y==max(y))[0][0]
      sigma_est = x0_est-np.where(y>0.68*np.array(max(y)))[0][0]   
      rms_list.append(np.std(baselines[:,freq]))

      try: 
        popt,pcov = curve_fit(gaussian,x,y,p0=[max(y),x0_est,sigma_est]) 
        x0_list.append(abs(popt[1]))
        sigma_list.append(abs(popt[2]))
      except:
        n_unfittable += 1
        logging.debug('Fit failed for frequency '+str(band_centers[freq])+'MHz.')
        #zero values will be stripped out in data cuts
        x0_list.append(0)
        sigma_list.append(0)

    x0 = np.array(x0_list)
    sigma = np.array(sigma_list)
    rms = np.array(rms_list)

    #data cuts
    rms_std = np.std(rms) #Remove frequencies with RMS greater than cutoff std dev's above mean.
    good_rms_flag = np.ndarray.tolist(np.where(rms<(cutoff*rms_std))[0])
    nonzero_x0_flag = np.ndarray.tolist(np.where(x0>0)[0])
    nonzero_sigma_flag = np.ndarray.tolist(np.where(sigma>0)[0])
    x0_sanity_flag = np.ndarray.tolist(np.where(x0<1024)[0])
    sigma_sanity_flag = np.ndarray.tolist(np.where(sigma<1024)[0])
    good_data = list(set(good_rms_flag) & set(nonzero_x0_flag) & set(nonzero_sigma_flag) & set(x0_sanity_flag) & set(sigma_sanity_flag)) 
    logging.info('Number of railed freqs: '+str(n_railed))
    logging.info('Number of unfittable freqs: '+str(n_unfittable))
    logging.info('Number of bad fits: '+str(1024-n_railed-len(set(nonzero_x0_flag) & set(nonzero_sigma_flag) & set(x0_sanity_flag) & set(sigma_sanity_flag))-n_unfittable))
    logging.info('Number of rms cuts: '+str(1024-len(good_rms_flag)))
    logging.info('Number of good freqs: '+str(len(good_data)))

    x0_med.append(np.median(x0[good_data]))
    sigma_med.append(np.median(sigma[good_data]))
    x0_uncert.append(np.std(x0[good_data]))
    sigma_uncert.append(np.std(sigma[good_data]))
    logging.info('x0 = '+str(x0_med[-1])+'+/-'+str(x0_uncert[-1]))
    logging.info('sigma = '+str(sigma_med[-1])+'+/-'+str(sigma_uncert[-1]))

    logging.info('Plotting...')
   
    #Plot data for individual baselines
    plot_transit_params_vs_freq(data_quality_dir, band_centers[good_data], unix_times, baseline_num, x0[good_data], sigma[good_data], rms[good_data], cutoff) 
    logging.info(baseline_num+' done!')
    logging.info(' ')

  #Plot comparisons across dishes
  plot_comparisons(data_quality_dir, unix_times, x0_med, x0_uncert, sigma_med, sigma_uncert)

  #Save test baseline and fit statistics for temporal stability analysis
  fit_stats = [iso_times, band_centers, baseline, x0_med, x0_uncert, sigma_med, sigma_uncert]
  stats_file = data_quality_dir+'/fit_stats_'+isot_times[0]+'.pkl'
  f = open(stats_file, 'wb')
  pickle.dump(fit_stats, f)
  f.close()

  #Close file once data is loaded 
  data_file.close()
  logging.info('Data file closed.')

########################################




########################################
def gaussian(x,a,x0,sigma):
  return a*np.exp(-(x-x0)**2/(2*(sigma)**2))
########################################



########################################
#Plot transit visibilities for a single baseline and freq
def plot_transit(data_quality_dir, unix_times, baseline_num, band_center, popt, x, y):

  #Set up data for plot
  datetimes = Time(unix_times, format='unix').datetime
  iso_times = Time(unix_times, format='unix').iso
  isot_times = Time(unix_times, format='unix').isot
  fit_to_plot = gaussian(x,*popt)

  #Make plot directory
  baseline_dir = data_quality_dir+'/baseline_'+baseline_num
  if not os.path.exists(baseline_dir):
    os.mkdir(baseline_dir)
    logging.info('Making baseline directory...')

  #Specify plot details
  fig, axis = plt.subplots()
  axis.set_title('Transit for Baseline '+baseline_num+' at '+band_center+' MHz')
  _ = axis.plot(x,y, 'k', label='data')
  _ = axis.plot(x,fit_to_plot, 'b', label='fit')
  axis.legend()
  axis.set_xlabel('UTC')
  _ = axis.set_xticks(x[0::100], minor=False)
  short_times = []
  for n in range(len(iso_times)):
    short_times.append(iso_times[n].split(' ')[1][:-7])
  tick_times = short_times[0::100]
  _ = axis.set_xticklabels(tick_times, rotation=90, minor=False)

  # Annotate with date of observation start
  axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
  axis.set_ylabel('Amplitude')
  plt.tight_layout()

  #Save and close figure
  plt.savefig(baseline_dir+'/transit_'+baseline_num+'_'+band_center+'MHz_'+isot_times[0]+'.png')    
  plt.close()
  
########################################




########################################
#Plot transit fit parameters (x0, sigma, fwhm) as a function of frequency
def plot_transit_params_vs_freq(data_quality_dir, band_centers, unix_times, baseline_num, x0, sigma, rms, cutoff):
  #Set up data for plot
  iso_times = Time(unix_times, format='unix').iso
  isot_times = Time(unix_times, format='unix').isot
  fwhm = 2*np.sqrt(2*np.log(2))*np.array(sigma)

  #Make plot directory
  baseline_dir = data_quality_dir+'/baseline_'+baseline_num
  if not os.path.exists(baseline_dir):
    os.mkdir(baseline_dir)
    logging.info('Making baseline directory...')

  #Plot x0 vs freq
  fig, axis = plt.subplots()
  axis.set_title('Transit Gaussian Fit for Baseline '+baseline_num)
  _ = axis.plot(band_centers,x0,'k.')
  # Annotate with date of observation start
  axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
  axis.set_xlabel('Frequency [MHz]')
  axis.set_ylabel('Transit peak time (UTC)')
  _ = axis.set_yticks(x0[0::100], minor=False)
  short_times = []
  for n in range(len(iso_times)):
    short_times.append(iso_times[n].split(' ')[1][:-7])
  tick_times = short_times[0::100]
  _ = axis.set_yticklabels(tick_times, minor=False)
  plt.tight_layout()

  #Save and close figure
  plt.savefig(baseline_dir+'/transit_x0_'+baseline_num+'_'+isot_times[0]+'.png')    
  plt.close()

  #Plot sigma vs freq
  fig, axis = plt.subplots()
  axis.set_title('Transit Gaussian Fit for Baseline '+baseline_num)
  _ = axis.plot(band_centers,sigma,'k.')
  axis.set_ylabel('Transit width /sigma [degrees]')
  # Annotate with date of observation start
  axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
  axis.set_xlabel('Frequency [MHz]')
  plt.tight_layout()

  #Save and close figure
  plt.savefig(baseline_dir+'/transit_sigma_'+baseline_num+'_'+isot_times[0]+'.png')    
  plt.close()

  #Plot FWHM vs freq
  fig, axis = plt.subplots()
  axis.set_title('Transit Gaussian Fit for Baseline '+baseline_num)
  _ = axis.plot(band_centers,fwhm,'k.')
  axis.set_ylabel('Transit FWHM [degrees]')
  # Annotate with date of observation start
  axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
  axis.set_xlabel('Frequency [MHz]')
  plt.tight_layout()

  #Save and close figure
  plt.savefig(baseline_dir+'/transit_fwhm_'+baseline_num+'_'+isot_times[0]+'.png')    
  plt.close()

  #Plot rms vs freq
  fig, axis = plt.subplots()
  axis.set_title('Transit Gaussian Fit for Baseline '+baseline_num)
  _ = axis.plot(band_centers,rms,'k.')
  axis.set_ylabel('Data RMS')
  # Annotate with date of observation start
  axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
  axis.set_xlabel('Frequency [MHz]')
  plt.tight_layout()

  #Save and close figure
  plt.savefig(baseline_dir+'/rms_'+baseline_num+'_'+isot_times[0]+'.png')    
  plt.close()

########################################



########################################
#Plot comparisons of transit parameters between dishes and polarizations
def plot_comparisons(data_quality_dir, unix_times, x0, x0_uncert, sigma, sigma_uncert):
  #Set up data for plot
  iso_times = Time(unix_times, format='unix').iso
  isot_times = Time(unix_times, format='unix').isot
  fwhm = 2.*np.sqrt(2.*np.log(2.))*np.array(sigma)
  fwhm_uncert = 2.*np.sqrt(2.*np.log(2.))*np.array(sigma_uncert)

  #Plot x0 comparison
  fig, axis = plt.subplots()
  axis.set_title('Transit Center Comparison')
  _ = axis.errorbar(range(len(x0)),x0,yerr=x0_uncert,fmt='ok')
  axis.set_xlabel('Correlator Channel')
  axis.set_ylabel('Transit peak time (UTC)')
  _ = axis.set_yticks(x0[0::10], minor=False)
  short_times = []
  for n in range(len(iso_times)):
    short_times.append(iso_times[n].split(' ')[1][:-7])
  tick_times = short_times[0::10]
  _ = axis.set_yticklabels(tick_times, minor=False)
  axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
  plt.tight_layout()

  #Save and close figure
  plt.savefig(data_quality_dir+'/transit_x0_comparison_'+isot_times[0]+'.png')    
  plt.close()

  #Plot sigma comparison
  fig, axis = plt.subplots()
  axis.set_title('Transit Width (Sigma) Comparison')
  _ = axis.errorbar(range(len(x0)),sigma,yerr=sigma_uncert,fmt='ok')
  axis.set_ylabel('Transit width /sigma [degrees]')
  axis.set_xlabel('Correlator Channel')
  axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
  plt.tight_layout()

  #Save and close figure
  plt.savefig(data_quality_dir+'/transit_sigma_comparison_'+isot_times[0]+'.png')    
  plt.close()

  #Plot FWHM comparison
  fig, axis = plt.subplots()
  axis.set_title('Transit Width (FWHM) Comparison')
  _ = axis.errorbar(range(len(x0)),fwhm,yerr=fwhm_uncert,fmt='ok')
  axis.set_ylabel('Transit FWHM [degrees]')
  axis.set_xlabel('Correlator Channel')
  axis.text(-0.015, 1.02, iso_times[0].split(' ')[0], ha='right', va='bottom', transform=axis.transAxes)
  plt.tight_layout()

  #Save and close figure
  plt.savefig(data_quality_dir+'/transit_fwhm_comparison_'+isot_times[0]+'.png')    
  plt.close()

########################################


if __name__ == '__main__':
  fit_transit_autocorr()

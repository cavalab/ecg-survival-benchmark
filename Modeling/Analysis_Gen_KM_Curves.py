# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:44:10 2024

"""

import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# C:\Users\ch242985\Desktop\Local ECG work\Trained_Models\MIMICIV\RibeiroReg_Best1YrMIMIC\EVAL


Model_One_Path = os.path.join(os.getcwd(), 'Trained_Models', 'MIMICIV', 'RibeiroReg_Best1YrMIMIC', 'EVAL', 'MIMICIV Test_Folder')
KM_Data_path = os.path.join(Model_One_Path, 'KM vs Model 100xBS Survival Curve.csv')
Hist_Path = os.path.join(Model_One_Path, 'Histogram.csv')

# with open(KM_Data_path)
KM_Data = np.genfromtxt(KM_Data_path, dtype = float, skip_header=1, delimiter=',')
hist_Data = np.genfromtxt(Hist_Path, dtype= float, skip_header=1, delimiter=',')

cuts = hist_Data[:,0] # end of the time bins
hist_coutns = hist_Data[:,1]
num_risk = hist_Data[:,2]

mdl_median  = KM_Data[:, 0]
mdl_int_low = KM_Data[:, 1]
mdl_int_high= KM_Data[:, 2]
km_median   = KM_Data[:, 3]
km_int_low  = KM_Data[:, 4]
km_int_high = KM_Data[:, 5]


plt.figure(figsize=(3.5, 8))
ax1 = plt.subplot(10,2,(1,16))
ax2 = plt.subplot(10,2,(19,20))

# fig1, ax = plt.savefig(args, kwargs)ubplots()
ax1.plot(cuts, km_median)
ax1.fill_between(cuts, km_int_low, km_int_high, color='b', alpha=.1)
ax1.plot(cuts,mdl_median, color='r')
ax1.fill_between(cuts, mdl_int_low, mdl_int_high, color='r', alpha=.1)
ax1.legend(('KM Median','KM 5th-95th%','Model Median','Model 5th-95th%'))
# plt.xlabel('Years')
# plt.ylabel('Survival')
ax1.set(ylabel = 'Survival' ,title = 'Model, Kaplan-Meier Survival(100 bootstraps)')


plt.text(-1, 400, 'Enrolled',color='r')

start = -1
end = 12
interval = (end - start) / 4

plt.text(start, 500, str(int(num_risk[0])),color='r')
plt.text(start + 1*interval, 500, str(int(num_risk[24])),color='r')
plt.text(start + 2*interval, 500, str(int(num_risk[49])),color='r')
plt.text(start + 3*interval, 500, str(int(num_risk[74])),color='r')
plt.text(start + 4*interval, 500, str(int(num_risk[99])),color='r')


# quant, bin_loc = np.histogram(cuts[disc_y_t],bins=surv.shape[1])
ax2.bar(cuts,hist_coutns,width= (max(cuts)-min(cuts))/len(cuts))
ax2.set(xlabel = 'Time to event or censor (years)' , ylabel = 'Count' )

# plot_file_path = os.path.join(temp_path, 'Time Dist Histogram.pdf')
# fig1.savefig(plot_file_path)

plot_file_path = os.path.join(os.getcwd(), 'Out_Figures', 'KM vs Model 100xBS Survival Curve.pdf')
plt.savefig(plot_file_path)
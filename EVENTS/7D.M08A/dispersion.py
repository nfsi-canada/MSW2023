from FTAN_functions import window, FTAN, GetDispersionCurve, smooth2, interpolate_disp_curve
from obspy import read
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read group dispersion for PREM
df = pd.read_csv('GDM52_dispersion.out', skiprows=5, names=['f', 'c_prem', 'c', 'dc', 'U_prem', 'U', 'dU'])
f_prem = df.f
U_prem = df.U_prem
U_GDM52 = df.U

# Read vertical seismograms
stream_raw = read('2012.069.07.09.BHZ.SAC')[0]
stream_corrected = read('CORRECTED/7D.M08A.2012.069.07.09.day.ZP-21.BHZ.SAC')[0]

# Time axis information
dt = stream_raw.stats.delta
nn = len(stream_raw.data)
time = np.arange(0, nn*dt, dt)

# Get distance between earthquake and station
distance_km = stream_raw.stats.sac.dist

# Get tmin and tmax based on maximum and minimum SW velocities expected
vmax = 5.5
vmin = 2.5
tmin = distance_km/vmax
tmax = distance_km/vmin

# Define frequencies at which to compute FTAN
periods = np.arange(10., 300.)
frequencies = 1./periods[::-1] # Strictly increasing frequencies

# Define velocities at which to compute FTAN
velocities = np.arange(2.5, 5., 0.01)

# Window data and calculate FTAN for both raw and corrected waveform
windowed_raw = window(stream_raw.data, time, dt, tmin, tmax)
amp_raw, freq_corrected = FTAN(windowed_raw, time, dt, 9660, frequencies, velocities, 50.)
amp_raw = smooth2(amp_raw)

windowed_corrected = window(stream_corrected.data, time, dt, tmin, tmax)
amp_corrected, freq_corrected = FTAN(windowed_corrected, time, dt, 9660, frequencies, velocities, 50.)
amp_corrected = smooth2(amp_corrected)

# Get dispersion curve and interpolate it
fcurve_raw, vcurve_raw, a_ = GetDispersionCurve(frequencies, velocities, amp_raw)
fcurve_raw_int, vcurve_raw_int = interpolate_disp_curve(frequencies, fcurve_raw[0], vcurve_raw[0])
fcurve_corrected, vcurve_corrected, a_ = GetDispersionCurve(frequencies, velocities, amp_corrected)
fcurve_corrected_int, vcurve_corrected_int = interpolate_disp_curve(frequencies, fcurve_corrected[0], vcurve_corrected[0])

# Make the figure
periods = 1./frequencies
X, Y = np.meshgrid(velocities, periods)

f = plt.figure(figsize=(8, 3.5))
plt.subplot(121)
cax = plt.pcolormesh(Y.T, X.T, amp_raw.T, shading='nearest')
plt.plot(1./fcurve_raw_int, vcurve_raw_int, 'r-', lw=2.5, label='Observed')
plt.plot(1./f_prem*1000., U_prem, 'w:', label='PREM')
plt.plot(1./f_prem*1000., U_GDM52, 'w-.', label='GDM52')

plt.xscale('log')
plt.colorbar(cax)
plt.ylabel('Group velocity (km/s)')
plt.xlabel('Period (s)')
plt.title('Raw vertical')
plt.legend(fontsize=8)

plt.subplot(122)
cax = plt.pcolormesh(Y.T, X.T, amp_corrected.T, shading='nearest')
plt.plot(1./fcurve_corrected_int, vcurve_corrected_int, 'r-', lw=2.5, label='Observed')
plt.plot(1./f_prem*1000., U_prem, 'w:', label='PREM')
plt.plot(1./f_prem*1000., U_GDM52, 'w--', label='GDM52')

plt.xscale('log')
plt.colorbar(cax)
# plt.ylabel('Group velocity (km/s)')
plt.xlabel('Period (s)')
plt.title('Corrected vertical')
plt.legend(fontsize=8)

plt.tight_layout()
plt.show()

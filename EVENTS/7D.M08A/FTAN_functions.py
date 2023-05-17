# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:16:45 2020

@author: david
"""

"""
Functions for Frequency-Time Analysis (FTAN) of surface waves.
"""

import copy
import numpy as np
from scipy.signal import hann,argrelextrema
from scipy.interpolate import RectBivariateSpline,interp1d
from scipy.optimize import minimize
from scipy import integrate
import obspy.signal.filter as opfilt
from obspy.signal.invsim import cosine_taper

def symmetrize(x_in):
    #Average the causal and acausal sides of a cross-correlation function.
    #Arguments:
    # x_in = a 1D array of a cross-correlation function
    #Returns
    # x = the symmetrized cross-correlation function
    
    x = copy.copy(x_in)
    midpt = int((len(x)-1)/2) #It would be 1 plus this, but python starts with 0.
    x = (x[midpt:] + x[midpt::-1])/2.
    return x

def window(x_in,t,dt,tmin,tmax,p=0.25,cut_left=False,cut_right=False):
    #Window the signal within the specified range of times.
    #The window is flat within the range [tmin,tmax] and hanning tapered at the edges outside of that.
    #Arguments:
    # x_in = the original signal (1D numpy array)
    # t = the times corresponding to each point in x_in (1D numpy array)
    # dt = time step between points in t
    # tmin = time at which left edge of taper begins
    # tmax = time at which right edge of taper begins
    # p = the percent of the range tmax-tmin to use as the width of the taper (on each side).
    # cut_left = tells whether or not to cut out the part of the signal to the left of the window.
    # cut_right = tells whether or not to cut out the part of the signal to the right of the window.
    #Returns:
    # x = the windowed signal (1D numpy array)
    
    x = copy.copy(x_in) #This is to avoid modifying the input x.
    taperlength = int(p*(tmax-tmin)/dt) #Length of the tapered part in number of samples.
    tapertime = dt*taperlength #Length of the tapered part in time, rounded to be a multiple of dt.
    tmin_round = np.round(tmin/dt)*dt
    tmax_round = np.round(tmax/dt)*dt
    tmin_taper = tmin_round-tapertime;
    tmax_taper = tmax_round+tapertime;
    hann_window = hann(2*taperlength+1)
    window = np.zeros(shape=t.shape)
    l_taper_mask = np.logical_and(t>=tmin_taper,t<=tmin_round)
    r_taper_mask = np.logical_and(t>=tmax_round,t<=tmax_taper)
    window[l_taper_mask]=hann_window[-(np.sum(l_taper_mask)-(taperlength+1)):taperlength+1] #The window on the left side is likely to bump up against t=0, so we have to avoid that.
    window[r_taper_mask]=hann_window[taperlength:taperlength+np.sum(r_taper_mask)] #The window on the right side is unlikely to bump up against t=tmax, but we're being careful anyway.
    window[np.logical_and(t>tmin_round,t<tmax_round)] = 1
    x *= window #Window the data
    if cut_right:
        x = x[t<=tmax_taper+dt/2] #The +dt/2 is to deal with problems that can otherwise arise due to floating point rounding errors.
        t = t[t<=tmax_taper+dt/2]
    if cut_left:
        x = x[t>=tmin_taper-dt/2]
    return x


def window2(x_in,t_in,dt,twindow,symmetric=False,cut_left=False,cut_right=False,cut_t_l=None,cut_t_r=None):
    #Window the signal within the specified range of times.
    #The window is flat within the range [twindow[1],twindow[1]] and hanning 
    #tapered from twindow[0] to twindow[1] and twindow[2] to twindow[2].
    #If times are kept as floating point numbers, small errors in those mess up
    #the comparisons, so I am converting all the times to integers before comparing
    #them.
    # x_in = the original signal (1D numpy array)
    # t = the times corresponding to each point in x_in (1D numpy array)
    # dt = time step between points in t
    # twindow = 4-element tuple of times defining the corners of the window.
    # symmetric = If true, uses absolute value of t to make the window, so can window both sides of a cross-correlation.
    # cut_left = tells whether or not to cut out the part of the signal to the left of the window.
    # cut_right = tells whether or not to cut out the part of the signal to the right of the window.
    # cut_t_l = If not None, gives a time to cut to on the left side if cut_left is True. If None, cuts to the edge of the window.
    # cut_t_r = If not None, gives a time to cut to on the right side if cut right is True. If None, cuts to the edge of the window.
    #Returns:
    # x = the windowed signal (1D numpy array)
    x = copy.copy(x_in) #This is to avoid modifying the input x.
    t = np.round(copy.copy(t_in)/dt).astype('int')
    tmin_round = int(np.round(twindow[1]/dt)) #Doing round before converting to int avoids problems due to floating point rounding errors.
    tmax_round = int(np.round(twindow[2]/dt))
    tmin_taper = int(np.round(twindow[0]/dt))
    tmax_taper = int(np.round(twindow[3]/dt))
    taperlength1 = int(np.round(tmin_round-tmin_taper))
    taperlength2 = int(np.round(tmax_taper-tmax_round))
    hann_window1 = hann(2*taperlength1+1)
    hann_window2 = hann(2*taperlength2+1)    
    window = np.zeros(shape=t.shape)
    if not symmetric:
        if tmin_taper<t.min() or tmax_taper>t.max():
            print('Error: Window edges should not be less than minimum t or greather than maximum t.')
        l_taper_mask = np.logical_and(t>=tmin_taper,t<=tmin_round)
        r_taper_mask = np.logical_and(t>=tmax_round,t<=tmax_taper)
        window[l_taper_mask] = hann_window1[:taperlength1+1]
        window[r_taper_mask] = hann_window2[taperlength2:]
        window[np.logical_and(t>tmin_round,t<tmax_round)] = 1
    else:
        if tmin_taper<0 or tmax_taper>t.max():
            print('Error: Window edges should not be less than 0 or greather than maximum t.')
        l_taper_mask = np.logical_and(np.abs(t)>=tmin_taper,np.abs(t)<=tmin_round)
        r_taper_mask = np.logical_and(np.abs(t)>=tmax_round,np.abs(t)<=tmax_taper)
        if t[l_taper_mask].size%2 == 0: #l_taper_mask will be odd if it includes 0 but even otherwise.
            window[l_taper_mask] = np.concatenate((hann_window1[taperlength1:],hann_window1[:taperlength1+1]))
        else: #It includes t = 0, so that is essentially a part of both sides.
            window[l_taper_mask] = np.concatenate((hann_window1[taperlength1:],hann_window1[1:taperlength1+1]))
        #print(tmax_round,tmax_taper,taperlength2)
        #print(t[r_taper_mask])
        window[r_taper_mask] = np.concatenate((hann_window2[:taperlength2+1],hann_window2[taperlength2:]))
        window[np.logical_and(np.abs(t)>tmin_round,np.abs(t)<tmax_round)] = 1
    x *= window #Window the data
    if cut_right:
        if cut_t_r is None:
            cut_t_r = tmax_taper
        else:
            cut_t_r = int(np.round(cut_t_r/dt)) #Convert to integer time.
        x = x[t<=cut_t_r]
        t = t[t<=cut_t_r]
    if cut_left:
        if cut_t_l is None:
            if not symmetric:
                cut_t_l = tmin_taper
            else:
                cut_t_l = -tmax_taper
        else:
            cut_t_l = int(np.round(cut_t_l/dt)) #Convert to integer time.
        x = x[t>=cut_t_l]
    return x

def FTAN(x,t,dt,dist,frequencies,velocities,params,normalize=True,transform ='gaus'):
    #This is a version of FTAN that gives allows Gaussian filters or the modified S-transform to be used.
    #Perform FTAN analysis and return an array of amplitudes at specified frequencies and velocities.
    #Arguments:
    # x = the data array (1D)
    # t = the times array corresponding to each point in x (must be increasing with constant step size dt and t[0]=0)
    # dt = time between samples in x
    # dist = source-receiver distance (distance between stations for ambient noise)
    # frequencies = array of frequencies at which to perform FTAN
    # velocities = array of velocities at which to return the FTAN amplitude
    # params = float or list, for Gaussian transform: width factor alpha; for modified S-transform: coefficients A and B 
    # normalize = boolean, tells whether or not to normalize the amplitude array so it is in the range [0,1]
    # transform = 'gaus' or 'mst', tells whether to use a Gaussian window or the modified S-transform.
    #Returns:
    # amplitude_resampled = a 2D numpy array of amplitudes at the specified frequencies and velocities
    # inst_freq_resampled = a 2D numpy array of instantaneous frequencies at the specified frequencies and velocities
    
    if transform == 'gaus':
        alpha = params
    elif transform == 'mst':
        A = params[0]
        B = params[1]
    else:
        print('Error: Unrecognized transform.')
    X = np.array([np.fft.fft(x) for Null in range(len(frequencies))])    # Fourier transform
    freq = np.fft.fftfreq(len(x), d=dt)
    amplitude = np.zeros(shape=(len(frequencies), len(x)))
    inst_freq = np.zeros(shape=(len(frequencies), len(x)))
    X_f0 = np.zeros(shape=X.shape,dtype=complex)
    for ifreq, f0 in enumerate(frequencies):     # Apply filters. Can I vectorize this loop?
        if transform == 'gaus':
            filt = np.exp(-alpha * ((freq - f0) / f0) ** 2)
        else:
            filt = np.exp(-2*(np.pi*(freq-f0)/(A*f0+B))**2)
        X_f0[ifreq,:] = X[ifreq,:] * (1.+np.sign(freq)) * filt #Equation 5.15 of Ritzwoller and Feng
    x_f0 = np.fft.ifft(X_f0,axis=1)
    amplitude = np.abs(x_f0)
    deriv = np.fft.ifft(X_f0*1j*2*np.pi*freq,axis=1) #dx_f0/dt Do the derivative in the frequency domain to try and reduce aliasing issues.
    numerator = smooth2(np.conj(x_f0)*deriv) #See the user manual to Chuck's code for a description of the algorithm used here to calculate instantaneous frequency.
    denominator = smooth2(np.conj(x_f0)*x_f0)
    inst_freq = np.abs(np.imag(numerator/denominator))/(2*np.pi)
    y = (dist/t[t>0])[::-1]
    z_amp = amplitude[:, t>0][:, ::-1]
    z_inst_freq = inst_freq[:, t>0][:, ::-1]
    amp_interpolant = RectBivariateSpline(frequencies, y, z_amp)
    amplitude_resampled = amp_interpolant(frequencies, velocities)
    amplitude_resampled[amplitude_resampled<0] = 0 #Fix the slightly negative amplitudes that can sometimes occur but shouldn't.
    inst_freq_interpolant = RectBivariateSpline(frequencies, y, z_inst_freq)
    inst_freq_resampled = inst_freq_interpolant(frequencies, velocities)
    if normalize:
            amplitude_resampled = amplitude_resampled/amplitude_resampled.max() #Normalize.
    return amplitude_resampled,inst_freq_resampled

def smooth2(x_in):
    #1-2-1 smoothing filter, similar to Chuck's code for this.
    #This version works along the second (horizontal) dimension of a 2D array.
    #Arguments:
    # x_in = the original signal (2D numpy array)
    #Returns:
    # x = the smoothed signal (2D numpy array)
    
    x = copy.copy(x_in) #To avoid modifying the original.
    n = x.shape[1]
    x[:,0] = 0.25*(x[:,1] + 2*x[:,0] + x[:,1])
    for i in range(1,n-1):
        x[:,i] = 0.25*(x[:,i-1] + 2*x[:,i] + x[:,i+1])
    x[:,n-1] = 0.25*(x[:,n-2] + x[:,n-2] + 2*x[:,n-1])
    return x

def GetDispersionCurve(frequencies,velocities,amplitude,nreturn=1,minamp=0.3,minampfrac_f=0.3,maxfjump=20,
                        maxvjump=20,maxslope=None,maxrevslope=None,allowoverlaps=True):
    #Extract the dispersion curves. This builds up curves by first finding a global
    #maximum and then working outwards from there in both directions to find amplitude
    #maxima at each frequency, choosing the maximum that is closest to the curve
    #in velocity. If multiple points are equally close, the one with higher amplitude
    #is chosen. If no points are within the allowed criteria, a new dispersion 
    #curve is started. The process is repeated nreturn times. There are two 
    #amplitude-based criteria for choosing which maxima to consider (minamp and 
    #minampfrac_f) and four ways of constraining the shape of the curve 
    #(maxfjump, maxvjump, maxslope, and maxrevslope).
    #Note: If there are plaeau minima, this may mistakenly count them as maxima.
    #It doesn't look like they are common, but it is something to watch out for.
    #Arguments:
    # frequencies = the frequencies corresponding to the first axis of the amplitudes array (1D array)
    # velocities = the velocities corresponding to the second axis of the amplitudes array (1D array)
    # amplitude = the amplitudes at the specified frequencies and velocities (2D array, shape = (len(frequencies),len(velocities))). All should be positive.
    # nreturn = number of curves to return (starting with the best).
    # minamp = The minimum absolute amplitude that is considered in extracting dispersion curves. Anything less than this is ignored, even if it forms a local maximum.
    # minampfrac_f = The fraction of the maximum amplitude for each frequency to consider in finding local maxima to expand each curve.
    # maxfjump = Maximum allowed frequency jump for crossing spectral holes
    # maxvjump = Maximum allowed change in velocity between any two consecutive points in a curve (regardless of slope or whether the points are at adjacent frequencies)
    # maxslope = Maximum allowed dispersion curve slope (change in velocity per change in frequency, units of m/s/Hz)
    # maxrevslope = Maximum allowed slope for reverse dispersion (increasing velocity with frequency) (change in velocity per change in frequency, units of m/s/Hz)
    # allowoverlaps = Whether or not to allow dispersion curves to overlap (boolean).
    #Returns:
    # fcurve_out = list of nreturn numpy arrays with frequencies of dispersion curves.
    # vcurve_out = list of nreturn numpy arrays with velocities of dispersion curves.
    # acurve_out = list of nreturn numpy arrays with amplitudes of dispersion curves.

    if maxfjump is None:
        maxfjump = frequencies.max()-frequencies.min()
    if maxvjump is None:
        maxvjump = velocities.max()-velocities.min()
    if maxslope is None:
        maxslope = np.inf
    if maxrevslope is None:
        maxrevslope = np.inf
    amp = copy.copy(amplitude) #Because just setting it equal only refers and doesn't really copy.
    #Find peaks in the amplitude matrix.
    freq_peaks_mask = np.zeros(amp.shape,dtype=bool) #Mask for peaks in the frequency direction.
    vel_peaks_mask = np.zeros(amp.shape,dtype=bool) #Mask for peaks in the velocity direction.
    freq_peaks_mask[argrelextrema(amp,np.greater_equal,axis=0)] = True #Maxima in the frequency direction
    vel_peaks_mask[argrelextrema(amp,np.greater_equal,axis=1)] = True #Maxima in the velocity direction
    freq_peaks_mask[np.logical_or(amp<minamp,amp<=0)] = False #Mask out peaks with too low amplitudes. The reason for the <=0 condition is to remove 0 plateaus if minamp = 0.
    vel_peaks_mask[np.logical_or(amp<minamp,amp<=0)] = False #Mask out peaks with too low amplitudes. The reason for the <=0 condition is to remove 0 plateaus if minamp = 0.
    peaks_mask = np.logical_and(vel_peaks_mask,freq_peaks_mask) #Maxima in both directions.
    fcurves_out = []
    vcurves_out = []
    acurves_out = []
    for n in range(nreturn):
        amp_masked = amp*peaks_mask #Amplitudes array with only peaks in it.
        if amp_masked.max() > 0: #If all maxima have been used, then there are no more dispersion curves to find.
            max_inds = np.unravel_index(amp_masked.argmax(), amp.shape) #Find the global maximum.
            fcurve = np.array([frequencies[max_inds[0]]]) #Start the dispersion curve with this point.
            vcurve = np.array([velocities[max_inds[1]]])
            acurve = np.array([amp[max_inds]])
            peaks_mask[max_inds] = False #Set this point to False since it's been used now.
            if not allowoverlaps:
                vel_peaks_mask[max_inds] = False #Set this point to False since it's been used now.
            ranges = [range(max_inds[0]-1,-1,-1),range(max_inds[0]+1,amp.shape[0],1)]  #First work backwards toward lower frequencies, then forwards toward higher frequencies.
            for irng,rng in enumerate(ranges):
                if irng == 1: #Working towards lower frequencies finished, but that curve was backwards, so we need to flip it.
                    fcurve = np.flip(fcurve)
                    vcurve = np.flip(vcurve)
                    acurve = np.flip(acurve)
                for i in rng:
                    a = amp[i,:] #The amplitudes at this frequency.
                    peaks_inds = np.where(np.logical_and(vel_peaks_mask[i,:],a>=minampfrac_f*a.max()))[0]
                    if peaks_inds.size > 0:
                        fjump = frequencies[i]-fcurve[-1] #This is the same for all.
                        if np.abs(fjump)>maxfjump: #Time to stop this curve.
                            break
                        vjump = velocities[peaks_inds]-vcurve[-1]
                        slope = vjump/fjump
                        sort_inds = np.lexsort((1/a[peaks_inds],np.abs(vjump))) #Sort the peaks first by change in velocity and second by amplitude.
                        for ind in sort_inds: #Now that I have the option for different positive and negative maximum slope, I may have to check more than one point.
                            if np.abs(vjump[ind])<=maxvjump and np.abs(slope[ind])<=maxslope and (
                                    slope[ind]<0 or slope[ind]<maxrevslope): #In velocity vs. frequency space, normal dispersion i sa negative slope and reverse dispersion is a positive slope.
                                fcurve = np.append(fcurve,frequencies[i]) #Add this point to the dispersion curve.
                                vcurve = np.append(vcurve,velocities[peaks_inds[ind]])
                                acurve = np.append(acurve,a[peaks_inds[ind]])
                                peaks_mask[i,peaks_inds[ind]] = False #In case this was a 2D maximum, set it to False, since it's been used in a curve now.
                                if not allowoverlaps:
                                    vel_peaks_mask[i,peaks_inds[ind]] = False #Set it to False, since it's been used in a curve now.
                                break
        else:
            fcurve = np.array([])
            vcurve = np.array([])
            acurve = np.array([])
        fcurves_out.append(fcurve)
        vcurves_out.append(vcurve)
        acurves_out.append(acurve)
    return fcurves_out,vcurves_out,acurves_out

def interpolate_disp_curve(frequencies,fcurve,vcurve):
    #Interpolate a dispersion curve onto an array of frequencies.
    #Frequencies within the bounds of the curve will be determined by linear 
    #interpolation (this also fills any spectral holes in the original curve), 
    #while frequencies beyond the ends of the curve will be set to NaN.
    #Arguments:
    # frequencies = frequencies at which to interpolate (1D numpy array)
    # fcurve, vcurve = frequencies and velocities of the dispersion curve (1D numpy arrays or lists of such arrays)
    #Returns:
    # fcurve_interp = frequencies at which interpolation was done (=frequencies input)
    # vcurve_interp = interpolated velocities
    
    if not isinstance(fcurve,list): #fcurve and vcurve, should either both or none be lists. If that's not the case, there will be problems.
        fcurve_interp = copy.copy(frequencies)
        vcurve_interp = np.full(frequencies.shape,np.nan)
        if fcurve.size != 0: #Generally, this shouldn't happen, but it can occasionally in some bad cases.
            f_used = np.logical_and(frequencies>=min(fcurve),frequencies<=max(fcurve)) #Mask for the frequencies actually used in this curve.
            vcurve_interp[f_used] = np.interp(frequencies[f_used],fcurve,vcurve) #Interpolate missing values.
    else:
        fcurve_interp = []
        vcurve_interp = []
        for n in range(len(fcurve)):
            fcurve_interp.append(copy.copy(frequencies))
            vcurve_interp.append(np.full(frequencies.shape,np.nan))            
            if len(fcurve)!=0:
                f_used = np.logical_and(frequencies>=min(fcurve[n]),frequencies<=max(fcurve[n])) #Mask for the frequencies actually used in this curve.
                print('fcurve[n]', n, fcurve[n])
                print(f_used)
                print(frequencies[f_used])
                vcurve_interp[n][f_used] = np.interp(frequencies[f_used],fcurve,vcurve) #Interpolate missing values.
    return fcurve_interp,vcurve_interp

def GetDispCurveInstFreq(frequencies,velocities,fcurve,vcurve,inst_freq,interp=True):
    #Calculate instantaneous frequencies along the dispersion curve.
    #Arguments:
    # frequencies = the frequencies of the FTAN diagram (1D numpy array)
    # velocities = the velocities of the FTAN diagram (1D numpy array)
    # fcurve = the frequencies of the dispersion curve (1D numpy array)
    # vcurve = the velocities of the dispersion curve (1D numpy array)
    # inst_freq = the array of instantaneous frequencies produced by FTAN (2D numpy array)
    # interp = boolean, tells whether to reinterpolate velocities back onto the frequencies of fcurve or to return the instantaneous frequencies at the original velocities
    #Returns:
    # f_out = a list of frequencies (1D numpy array), either the original fcurve or the instantaneous frequencies
    # v_out = a list of velocities (1D numpy array), either the interpolated velocities along fcurve or the original velocities now at the corresponding instantaneous frequencies
    
    interpolant = RectBivariateSpline(frequencies,velocities,inst_freq) #If fcurve and vcurve are in frequencies and velocities, we don't really need to interpolate, but this approach is more general.
    f_inst_curve = interpolant(fcurve,vcurve,grid=False) #Evaluate the instantaneous frequence at the points along the dispersion curve.
    mask = ~np.isnan(f_inst_curve) #Ingore nans (which come from having nan velocities where the curve isn't defined).
    #if np.sum(mask)>1 and np.min(inst_freq[mask][1:len(inst_freq[mask])]-inst_freq[mask][0:len(inst_freq[mask])-1])<0:#Check if the curve if monotonically increasing, and if not fix it.
    if np.sum(mask)>1 and np.min(f_inst_curve[mask][1:len(f_inst_curve[mask])]-f_inst_curve[mask][0:len(f_inst_curve[mask])-1])<0:#Check if the curve if monotonically increasing, and if not fix it.    
    #Find the monotonically increasing curve that best fits the instantaneous periods.
        #This is modified from the approach taken by Goutorbe et al. (2015) for doing this.
        #This is rather slow, so consider finding a faster way to do it.
        def fun(freq):
            # misfit wrt calculated instantaneous periods
            return np.sum((freq - f_inst_curve[mask])**2)
        def con_fun(inst_freq_): #Constraing function, must be increasing
            minslope = np.min(inst_freq_[1:len(inst_freq_)]-inst_freq_[0:len(inst_freq_)-1])
            return minslope #If all our increasing, this will be positive and the constraint will be satisfied. But if any are decreasing, then it won't be.
        constraints = {'type': 'ineq', 'fun': con_fun}
        res = minimize(fun, x0=f_inst_curve[mask], method='SLSQP', constraints=constraints)
        f_inst_curve[mask] = res['x']
    if interp: #Interpolate back to the original frequencies.
        f_out = fcurve
        if np.sum(mask)>1: #Need at least two points to interpolate between
            v_out = np.interp(f_out,f_inst_curve[mask],vcurve[mask],left=np.nan,right=np.nan) #Note: np.interp will have a problem if the values of f_inst aren't monotonically increasing.
        else:
            v_out = np.full(f_out.shape,np.nan)
    else:
        f_out = f_inst_curve
        v_out = vcurve
    return f_out,v_out

def phase_matched_filter(frequencies,velocities,dist,x_in,t,dt,cut_snr=0.2):
    #Apply phase-matched filtering to clean data, following the method of Levshin and Ritzwoller (2001) section 3. (Also somewhat following parts of pysismo and aftan codes.)
    #Note: Currently input has to be in time domain, but I might want to allow it to be in frequency domain too to allow it to be called in FTAN after fft.
    #Arguments:
    # frequencies = list of frequencies for the initial dispersion curve used to define the phase-matched filters.
    # velocities = list of velocities for the initial dispersion curve, corresponding to fdispcurve. Use nan for an unknown velocity.
    # dist = distance between the stations (source-receiver distance).
    # x_in = the data array (1D). This should start at t = 0.
    # t = the times corresponding to each point in x_in (1D numpy array)
    # dt = time between samples in x    
    # cut_snr = ratio
    #Returns:
    # xc = The "cleaned" signal x with noise cut out.
    
    #Make a copy of x_in so we don't modify it:
    x = copy.copy(x_in)
    #Calculate the phase correction factors (psi) and make an interpolant to calculate them at any frequency:
    mask1 = ~np.isnan(velocities)
    if np.sum(mask1) > 0: #Make sure there are some velocities that aren't nan.
        k = np.zeros(shape=frequencies[mask1].shape) #Wavenumber, I think.
        k[0] = 0.0 #We don't really know k[0], but it doesn't matter according to Levshin and Ritzwoller (2001)
        k[1:] = 2 * np.pi * integrate.cumtrapz(1.0/velocities[mask1],x=frequencies[mask1]) #Eq. 3 of Levshin and Ritzwoller (2001)
        psi = k * dist #Phase correction, Eq. 4 of Levshin and Ritzwoller (2001)
        interpolant =  interp1d(frequencies[mask1],psi) #Use this later to interpolate phi as a function of frequency.
        #Apply the phase-matched filter in the frequency domain:
        X = np.fft.fft(x)    # Fourier transform. Should I double positive side and set negative to 0 or does that not matter?
        freq = np.fft.fftfreq(len(X), d=dt)
        mask2 = (freq >=frequencies[mask1].min()) & (freq <= frequencies[mask1].max())
        psi2 = interpolant(freq[mask2])
        X[mask2] = X[mask2] * np.exp(1j*psi2) #Should this be X or np.abs(X) as pysismo does? From Levshin papers, I don't think so. Also pysismo has -1j, but from Levshin et al. (1992) it looks like it should just be 1j.
        taper = cosine_taper(npts=mask2.sum(), p=0.05) #This follows pysismo. aftan has a more complicated tapering function, which I don't fully understand.
        X[mask2] *= taper
        X[~mask2] = 0. #Frequencies outside of the range covered by the dispersion curve.
        #Should I taper in the time domain? It looks like aftan does that whether or not it's doing phase-matched filtering.
        #Cut signal in the time domain: (Some of this is inspired by aftan, but that code is complicated and doesn't have a lot of comments, so I don't think this is quite the same.)
        x = np.fft.ifft(X) #Return to time domain.
        env = np.abs(x)
        env_max = env.max() #Maximum value of the envelope
        ind_max = np.argwhere(env==env_max) #Index at which the maximum occurs.
        if len(ind_max)>1:
            print('Warning: Multiple points at global maximum: using midpoint.')
            ind_max = ind_max[int(len(ind_max)/2)]
        min_inds = 1+np.argwhere(np.logical_and(env[0:len(env)-2]>env[1:len(env)-1],env[1:len(env)-1]<env[2:len(env)])) #Last and first points are left out; there's no point in cutting there anyway.
        cutind_l = 0 #Initialize these to the end of the waveform, so if no cutoff is found, it just keeps the whole waveform on that side.
        cutind_r = len(env)-1
        for n in min_inds[min_inds<ind_max][::-1]: #Cut the left side. (There may be a more efficient, vectorized way to do this.)
            if env[n] < cut_snr*env_max:
                cutind_l = n
                break
        for n in min_inds[min_inds>ind_max]:
            if env[n] < cut_snr*env_max:
                cutind_r = n
                break
        x = window(x,t,dt,t[cutind_l],t[cutind_r],0.10) #Taper off instead of cutting sharply, but do it fairly quickly.
        #Return to the frequency domain, redisperse, and then return to the time domain again:
        X = np.fft.fft(x)
        X[mask2] *= np.exp(-1j*psi2) #Remove the phase-matched filter.
        X[~mask2] = 0 #Should I do this? There shouldn't be much in here if there is anything.
        xc = np.fft.ifft(X) #Final, cleaned signal.
    else: #If everything is nan, just 0 out this signal. There is nothing to see here.
        xc = np.zeros(shape=x.shape)
    return xc

def extrapolate_disp_curve(fcurve,vcurve_in,degree=0,left=False,right=True):
    #Extrapolate a dispersion curve to points beyond its left end.
    #vcurve must have points on its ends that are NaN, such as is output from 
    #interpolate_disp_curve or GetDispCurveInstFreq. It is these points that 
    #will have an extrapolated value assigned to them.
    #Arguments:
    # fcurve = a 1D numpy array of frequencies in ascending order
    # vcurve_in = a 1D numpy array of velocities at the frequencies in fcurve
    # degree = the degree of the extrapolating function. Only 0 or 1 are allowed.
    #          If 0, all the points beyond the edge of the curve are set to the 
    #          value of the last point; if 1, they are set to a line with the 
    #          slope defined by the last two points.
    # left = boolean, tells whether to extrapolate to the left of the curve (lower frequencies)
    # right = boolean, tells whether to extrapolate to the right of the curve (higher frequencies)
    #Returns:
    # vcurve = extrapolated velocities at frequencies in fcurve (1D numpy array)
    
    vcurve = copy.copy(vcurve_in) #Avoid changing inputs.
    if np.any(~np.isnan(vcurve)): #There needs to be at least one non-nan value.
        if left:
            indlast = int(np.argwhere(fcurve==min(fcurve[~np.isnan(vcurve)]))) #Lowest frequency with a known velocity.
            if degree==0:
                vcurve[0:indlast] = vcurve[indlast]
            elif degree==1:
                vslope = (vcurve[indlast]-vcurve[indlast+1])/(fcurve[indlast]-fcurve[indlast+1])
                vintercept = vcurve[indlast]-vslope*fcurve[indlast]
                vcurve[0:indlast] = vslope*fcurve[0:indlast]+vintercept
            else:
                print('Error: degree other than 0 or 1 is not allowed.')
        if right:
            indlast = int(np.argwhere(fcurve==max(fcurve[~np.isnan(vcurve)]))) #Lowest frequency with a known velocity.
            n = len(fcurve)
            if degree==0:
                vcurve[indlast+1:n] = vcurve[indlast]
            elif degree==1:
                vslope = (vcurve[indlast-1]-vcurve[indlast])/(fcurve[indlast-1]-fcurve[indlast])
                vintercept = vcurve[indlast]-vslope*fcurve[indlast]
                vcurve[indlast+1:n] = vslope*fcurve[indlast+1:n]+vintercept
            else:
                print('Error: degree other than 0 or 1 is not allowed.')
    else: #Nothing to do for this curve.
        print('All velocities are nan. Cannot extrapolate.')
    return vcurve

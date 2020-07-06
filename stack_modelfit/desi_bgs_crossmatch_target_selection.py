import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm 
from astropy.io import fits 
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import pandas as pd
from copy import deepcopy
from __future__ import division

max_radius = 9.6963e-6 # match radius to 2 arcseconds, this is in radians
max_radius_gaia = 4.84815e-6
ra_min = 0
ra_max = 3
dec_min = -3
dec_max = 0
cols=["ra","dec"]
string = str(ra_min) + '_' + str(ra_max) + '_' + str(dec_min) + '_' + str(dec_max)

# ---------------------------------------   

#based on Koposov 2017 cut, valid for between 15 < G < 19 I believe
def astrometric_excess_cut(mag):
    lim = 0.15*(mag-15) + 0.25
    e = 10**lim
    return e

def piecewise_linear(x, x0, y0, k1, k2):
    y = np.piecewise(x, [x < x0, x >= x0],
                     [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    return y

def cubic_fit(x, a, b, c, d):
    g = a*(x-2)**3 + b*(x-2)**2 + c*(x-2) + d
    return g

def exclusion_radius_gaia(mag):
    if mag < 9:
	    r = 708.9*np.exp(-mag/8.41)
    else:
	    r = 694.7*np.exp(-mag/4.04)
    return r


# exclusion radii from Hyper Suprime Cam, I end up using the elg exclusion radius as a more conservative
# measure, but this can certainly be modified. 
def exclusion_radius_elg(mag):
    r = 10**(3.5-0.15*mag)
    return r
def exclusion_radius_bgs(mag):
    r = 10**(2.2-0.15*mag)
    return r

def nanomaggy_to_mag(n):
    m = 22.5 - 2.5*np.log10(n)
    return m


def filter_decals_mags(decals, min_mag, max_mag):
    
    decals_filter_all = [[] for x in xrange(len(decals))]
    count = 0
    
    for i in range(len(decals[0])):
        rflux = decals[3][i][2]
        if rflux > 0:
            rmag = nanomaggy_to_mag(rflux)
            if rmag > min_mag:
                if rmag < max_mag:
                    for j in range(len(decals)):
                        decals_filter_all[j].append(decals[j][i])
                    count += 1
    
    print(str(count) + ' of ' + str(len(decals[0])) + ' sources had ' + str(min_mag) + ' < r < ' + str(max_mag))
    return decals_filter_all

# for my project specifically, where I restricted to 0 < RA < 3, -3 < DEC < 0
def restricted_area_catalog(ra_min, ra_max, dec_min, dec_max, decals_list_all):
    decals_info = [[] for x in xrange(len(decals_list_all))]
    for i in range(len(decals_list_all[0])):
        if decals_list_all[0][i] > ra_min and decals_list_all[0][i] < ra_max:
            if decals_list_all[1][i] > dec_min and decals_list_all[1][i] < dec_max:             
		        for j in range(len(decals_list_all)):
			        decals_info[j].append(decals_list_all[j][i])                

    print(str(len(decals_info[0])) + ' sources isolated from an original ' + str(len(decals_list_all[0])))
    return decals_info

# cross matches between specific catalogs, where indices 0, 1 of each source vector contain the position in radians 
def general_cross_match(tolerance, catalog1, catalog2, bit=11, nfalse=4):
    print('Length of first catalog: ' + str(len(catalog1)))
    print('Length of second catalog: ' + str(len(catalog2)))
 
    new_catalog = []
    ps1 = [[item[0], item[1]] for item in catalog1]
    ps2 = [[item[0], item[1]] for item in catalog2]
    v1len = []
    
    # positional cross match done here ---
    kdt = cKDTree(ps2)
    obj = kdt.query_ball_point(ps1, tolerance)
    
    zero_count, one_count, more_count = [0 for x in xrange(3)]

    for i in range(len(obj)):
        object_len = len(obj[i])
	    vector1 = list(catalog1[i])
	    v1len.append(len(vector1))
	    if object_len == 0:
	        for n in xrange(nfalse):
		        vector1.append(False)
	        zero_count += 1
	        new_catalog.append(vector1)
	    # if one match or more, take the first object matched to for counterpart in catalog
        elif object_len >= 1:
	        index = obj[i][0]
	        vec = catalog2[index]
	        for n in xrange(nfalse):
		        vector1.append(vec[2+n])
	        if object_len == 1:
		        vector1[bit] = True
		        one_count += 1
	        else:
		        vector1[bit] = object_len
		        more_count += 1
	        new_catalog.append(vector1)
    
    print(str(zero_count) + ' sources in first catalog had zero matches')
    print(str(one_count) + ' sources in first catalog had one match')
    print(str(more_count) + ' sources in first catalog had more than one match')
    
    return new_catalog   

def flag_near_brightstar(catalog_list, bright_star_catalog):
    new_catalog = []
    count = 0
    catalog_ps = [[i[0], i[1]] for i in catalog_list]
    bitarray = [False]*len(catalog_list)
    
    # positional cross match for each bright star with respect to catalog
    for source in bright_star_catalog:
	    pos = [source[0], source[1]]
	    kdt = cKDTree([pos])
    	obj = kdt.query_ball_point(catalog_ps, source[2])
	    # if a source is near a bright star, sit its bright star bit flag to True
	    for o in range(len(obj)):
	        if len(obj[o]) > 0:
		    bitarray[o] = True
	# append bright star bit flag to existing catalog
    for i in range(len(catalog_list)):
	    source = list(catalog_list[i]) + [bitarray[i]]
	    new_catalog.append(source)
	    if bitarray[i] > 0:
	        count += 1
    
    print(str(count) + ' sources were flagged near bright sources.')
    
    return new_catalog


def rwise_gz_fit(list1, list2, nsig=3, plot='yes', save='no', deg=3):
    gz = list1
    rwise = list2
    n=1
    it=0
    tot = 0

    #iterative sigma clipping, n represents sources that are excluded from each iteration
    while n > 0:
    	if deg==3:
	        popt, pcov = curve_fit(cubic_fit, gz, rwise)
	        predict = []
	        for x in gz:
		        predict.append(cubic_fit(x, *popt))
    	    p = np.poly1d(np.polyfit(gz, rwise, deg))
    	else:
	        best_fit_line = np.polyfit(gz, rwise, deg)
            p = np.poly1d(best_fit_line)
            predict = p(gz)
        rms = np.sqrt(np.sum((np.array(predict)-np.array(rwise))**2)/len(rwise))
        
        temp_gz, temp_rw = [[],[]]
        n=0
        for i in range(len(rwise)):
            if np.abs(predict[i]-rwise[i]) < nsig*rms:
                temp_gz.append(gz[i])
                temp_rw.append(rwise[i])
            else:
                n+=1
    	tot += n        
	    gz = temp_gz
        rwise = temp_rw
        it+=1
    print('After ' + str(it) + ' iterations, ' + str(tot) + ' sources clipped..')
    print('Final number of samples in fit: ' + str(len(gz)))
    
    if plot=='yes':
        colorspace = np.linspace(np.min(gz), np.max(gz), 100)
        plt.figure()
        plt.title(str(nsig) + '-Sigma Clipped Color-Color Diagram, D1719G-WISE, degree = ' + str(deg))
        plt.scatter(gz, rwise, s=5)
        plt.xlabel('g - z')
        plt.ylabel('r - W1')
        if deg==3:
	        plt.plot(colorspace, cubic_fit(colorspace, *popt), label='rms = ' + str(np.round(rms, 3)), color='g', linewidth=5)
	        print('Best fit function: ' + str(popt[0]) + '(x-2)**3 + ' + str(popt[1]) + '(x-2)**2 + ' + str(popt[2]) + '(x-2) + ' + str(popt[3]))
	    else:
	        plt.plot(colorspace, p(colorspace), label='rms = ' + str(np.round(rms, 3)), color='g', linewidth=5)
        plt.legend()
        if save=='yes':
	        plt.savefig('presentation_figures/D1719GW_' + str(nsig) + 'sigma_clip_gz_rw1_deg' + str(deg) + '.png')
        plt.show()
    
    return p



def gaiar_gz_fit(list1, list2, nsig=3, plot='no', save='no'):
    gz = list1
    gaiar = list2
    n=1
    it=0
    tot=0
    while n > 0:
        n=0
	    popt, pcov = curve_fit(piecewise_linear, gz, gaiar, p0=[2.1, 0.1, 0.01, -0.5]) # approximate first guess but result is more precise than the initial guess
	    predict_z = piecewise_linear(gz, *popt)
	    delta_gaia_z = predict_z - gaiar
	    rms = np.sqrt(np.sum((predict_z-gaiar)**2)/len(gz))

    	temp_gaiar, temp_gz = [[],[]]

    	for j in range(len(predict_z)):
	        if np.abs(delta_gaia_z[j]) < nsig*rms:
             	temp_gz.append(gz[j])
        	    temp_gaiar.append(gaiar[j])
	        else:
    		    n+=1
        tot += n        
    	gz = temp_gz
        gaiar = temp_gaiar
        it+=1    
    print('After ' + str(it) + ' iterations, ' + str(tot) + ' sources clipped..')
    print('Final number of samples in fit: ' + str(len(gz)))
    
    if plot=='yes':
        colorspace = np.linspace(np.min(gz), np.max(gz), 100)
        plt.figure()
        plt.title(str(nsig) + '-Sigma Clipped Color-Color Diagram, D1719G-WISE')
        plt.scatter(gz, gaiar)
        plt.xlabel('g - z')
        plt.ylabel('Gaia - r')
        plt.plot(colorspace, piecewise_linear(colorspace, *popt), label='rms = ' + str(np.round(rms, 3)), color='g', linewidth=5)
        plt.legend()
        if save=='yes':
            #plt.savefig(str(string) + '/DECaLS_17_19-Gaia/D1719G-WISE/' + str(nsig) + 'sigma_clip_gaiar_gz.png')
	        plt.savefig('presentation_figures/D1719GW_' + str(nsig) + 'sigma_clip_gaiar_gz.png')
        plt.show()
    
    return popt

       

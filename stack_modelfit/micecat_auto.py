import time
import os
import shutil
import numpy as np
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import bz2
from ciber_info import *
from srcmap import *
from stack_ancillary import *

def get_micecat_fields_coords():
    micecat_fields_coords = []

    count  = 0
    for dec_cent in range(1,90,2):
        dec_min = dec_cent - 1
        dec_max = dec_cent + 1
        ra_length = 90 * np.cos(dec_max * np.pi / 180)
        Nfields = int(np.round(ra_length / 2))
        ra_cents = np.linspace(1/np.cos(dec_max * np.pi / 180),
                             90 - 1/np.cos(dec_max * np.pi / 180), Nfields)
        for ra_cent in ra_cents:
            ra_max = ra_cent + 1/np.cos(dec_max * np.pi / 180)
            ra_min = ra_cent - 1/np.cos(dec_min * np.pi / 180)

            micecat_fields_coords.append((ra_cent, ra_min, ra_max,
                                         dec_cent, dec_min, dec_max))
            count += 1

    return micecat_fields_coords

def gen_sql(icat, select_top=None):
    micecat_fields_coords = get_micecat_fields_coords()
    ra_cent, ra_min, ra_max, dec_cent, dec_min, dec_max = micecat_fields_coords[icat]
    select_fields = ['unique_halo_id', 'flag_central','nsats', 'ra_gal', 'dec_gal',
              'z_cgal', 'lmhalo', 'lmstellar', 'lsfr',
              'euclid_nisp_y_true', 'euclid_nisp_j_true', 'euclid_nisp_h_true']
    if select_top is None:
        select_str = ''
    else:
        select_str = ' LIMIT %d'%select_top
    sql = 'SELECT ' + ','.join(f for f in select_fields) \
        + ' FROM micecatv2_0_view' \
        + ' WHERE ' + 'ra_gal>{}'.format(ra_min) \
        + ' AND ' + 'ra_gal<{}'.format(ra_max) \
        + ' AND ' + 'dec_gal>{}'.format(dec_min) \
        + ' AND ' + 'dec_gal<{}'.format(dec_max) \
        + select_str
    
    return sql
def query_micecat():
    driver = webdriver.Chrome(mypaths['ciberdir'] + 'python_ciber/' + 'chromedriver')
    driver.get('https://cosmohub.pic.es/login')

    wait_time = 0.1

    while True:
        try:
            id_box = driver.find_element_by_name('username')
            id_box.send_keys('ycheng3@caltech.edu')
            id_box = driver.find_element_by_name('password')
            id_box.send_keys('ytccosmohub')
            driver.find_element_by_xpath("//button[@type='submit']").click()
            break
        except NoSuchElementException:
            time.sleep(wait_time)

    while True:
        try:
            driver.find_element_by_xpath("//a[@href='/catalogs/1']").click()
            break
        except NoSuchElementException:
            time.sleep(wait_time)

    time.sleep(10)
    while True:
        try:
            driver.find_element_by_xpath("//button[@ng-disable='!query.readonly']").click()
            break
        except NoSuchElementException:
            time.sleep(wait_time)

    time.sleep(5)
    while True:
        try:
            driver.find_element_by_xpath("//button[@ng-click='query.readonly = !query.readonly']").click()
            break
        except NoSuchElementException:
            time.sleep(wait_time)
    
    return driver


def sendSQL_micecat(driver, sql):
    time.sleep(3)
    sql_box = driver.find_element_by_xpath("//textarea[@class='ace_text-input']")
    for i in range(1000):
        sql_box.send_keys(Keys.DELETE)
    for i in range(1000):
        sql_box.send_keys(Keys.BACK_SPACE)

    sql_box.send_keys(sql)
    time.sleep(1)
    driver.find_element_by_xpath("//div[@class='checkbox']").click()
    time.sleep(20)
    driver.find_element_by_xpath("//button[@id='requestButton']").click()
    time.sleep(10)
    while True:
        try:
            request_id = driver.find_element_by_xpath("//text[@ng-bind-html='request.message']").text
            
            if 'error' in request_id:
                print('REQUEST ERROR!!!')
                request_id = False
                break
            request_id = int(request_id.replace('!', ' ').split()[7])
            break
        except:
            time.sleep(0.1)
    
    time.sleep(5)
    
    return request_id

def dowload_micecat(request_id_dict):
    driver = webdriver.Chrome(mypaths['ciberdir'] + 'python_ciber/' + 'chromedriver')
    driver.get('https://cosmohub.pic.es/login')

    wait_time = 0.1

    while True:
        try:
            id_box = driver.find_element_by_name('username')
            id_box.send_keys('ycheng3@caltech.edu')
            id_box = driver.find_element_by_name('password')
            id_box.send_keys('ytccosmohub')
            driver.find_element_by_xpath("//button[@type='submit']").click()
            break
        except NoSuchElementException:
            time.sleep(wait_time)

    while True:
        try:
            driver.find_element_by_xpath("//a[@href='/activity']").click()
            break
        except NoSuchElementException:
            time.sleep(wait_time)    

    for icat, request_id in request_id_dict.items():            
        while True:
            try:
                driver.find_element_by_xpath("//a[contains(@href, %s)]"%(str(request_id))).click()
                break
            except NoSuchElementException:
                time.sleep(wait_time)
                
    return driver

def process_downloaded_micecat(request_id_dict):
    for icat, request_id in request_id_dict.items():  
        try:
            os.rename('/Users/ytcheng/Downloads/' + str(request_id) + '.csv.bz2',
                     '/Users/ytcheng/Downloads/micecat_%d.csv.bz2'%icat)
            shutil.move('/Users/ytcheng/Downloads/micecat_%d.csv.bz2'%icat,
                        mypaths['MCcatdat'] + 'all_fields/' + 'micecat_%d.csv.bz2'%icat)
        except FileNotFoundError:
            continue
    return

def run_micecat_query(icat_list):
    for icat in icat_list:
        driver = query_micecat()
        sql = gen_sql(icat, select_top=None)
        request_id = sendSQL_micecat(driver, sql)
        print('query cat %d, request id %d'%(icat, request_id))
        driver.quit()
        
        time.sleep(120)
        driver = dowload_micecat({icat: request_id})
        time.sleep(60)
        driver.quit()
        process_downloaded_micecat({icat: request_id})


def get_micecat_df_auto(icat, add_Rvir=False):
    fname = mypaths['MCcatdat'] + 'all_fields/' + 'micecat_%d.csv.bz2'%icat
    micecat_fields_coords = get_micecat_fields_coords()
    ra_cent, ra_min, ra_max, dec_cent, dec_min, dec_max = micecat_fields_coords[icat]
    with bz2.BZ2File(fname) as catalog_fd:
        df = pd.read_csv(catalog_fd, sep=",", index_col=False,
                         comment='#', na_values=r'\N')

    # apply mag correction, c.f. data query page
    # magnitude_evolved = magnitude_catalog - 0.8 * (atan(1.5 * z_cgal) - 0.1489)
    df['euclid_nisp_y_true'] = df['euclid_nisp_y_true'].values \
    - 0.8 * (np.arctan(1.5 * df['z_cgal'].values) - 0.1489)
    df['euclid_nisp_j_true'] = df['euclid_nisp_j_true'].values \
    - 0.8 * (np.arctan(1.5 * df['z_cgal'].values) - 0.1489)
    df['euclid_nisp_h_true'] = df['euclid_nisp_h_true'].values \
    - 0.8 * (np.arctan(1.5 * df['z_cgal'].values) - 0.1489)


    df['I'] = df['euclid_nisp_y_true']
    Hmag = 0.5*(10**(-df['euclid_nisp_j_true']/2.5) + 10**(-df['euclid_nisp_h_true']/2.5))
    Hmag = -2.5*np.log10(Hmag)
    df['H'] = Hmag

    df['x'] = (df['ra_gal'] - ra_cent)/(np.cos(df['dec_gal'] * np.pi / 180))*3600 / 7 + 511.5
    df['y'] = (df['dec_gal'] - dec_cent)*3600 / 7 + 511.5


    df.drop(['lsfr','lmstellar', 'ra_gal', 'dec_gal','euclid_nisp_y_true',
             'euclid_nisp_j_true','euclid_nisp_h_true'], axis=1, inplace=True)

    if add_Rvir:
        z_arr = np.array(df['z_cgal'])
        Mh_arr = 10**np.array(df['lmhalo'])
        rhoc_arr = np.array(cosmo.critical_density(z_arr).to(u.M_sun / u.Mpc**3))
        rvir_arr = ((3 * Mh_arr) / (4 * np.pi * 200 * rhoc_arr))**(1./3)
        DA_arr = np.array(cosmo.comoving_distance(z_arr))
        rvir_ang_arr = (rvir_arr / DA_arr) * u.rad.to(u.arcsec)
        df['Rv_Mpc'] = rvir_arr
        df['Rv_arcsec'] = rvir_ang_arr
        
    return df

def radial_binning(rbins,rbinedges):
    rsubbinedges = np.concatenate((rbinedges[:1],rbinedges[6:20],rbinedges[-1:]))

    # calculate 
    rin = (2./3) * (rsubbinedges[1]**3 - rsubbinedges[0]**3)\
    / (rsubbinedges[1]**2 - rsubbinedges[0]**2)

    rout = (2./3) * (rsubbinedges[-1]**3 - rsubbinedges[-2]**3)\
    / (rsubbinedges[-1]**2 - rsubbinedges[-2]**2)

    rsubbins = np.concatenate(([rin],rbins[6:19],[rout]))

    return rsubbins, rsubbinedges

def run_micecat_auto_fliter_test_cen(inst, icat, filt_order_arr=[0], mag_stack = [0,1],
                           savedir = './micecat_data/all_fields/', save_data = True):
    
    df = get_micecat_df_auto(icat)
    df = df[df['flag_central']==0]
    
    mag_th = 20
    xs, ys, ms = np.array(df['x']), np.array(df['y']), np.array(df['I'])
    ms_inband = np.array(df['I']) if inst==1 else np.array(df['H'])

    make_srcmap_class = make_srcmap(inst)
    spb = np.where(df['I']<=mag_th)[0]
    spf = np.where(df['I']>mag_th)[0]

    make_srcmap_class.ms = ms[spb]
    make_srcmap_class.ms_inband = ms_inband[spb]
    make_srcmap_class.xls = xs[spb]
    make_srcmap_class.yls = ys[spb]
    srcmapb = make_srcmap_class.run_srcmap(ptsrc=True, verbose=False)

    make_srcmap_class.ms = ms[spf]
    make_srcmap_class.ms_inband = ms[spf]
    make_srcmap_class.xls = xs[spf]
    make_srcmap_class.yls = ys[spf]
    srcmapf = make_srcmap_class.run_srcmap_nopsf()
    srcmap = srcmapb + srcmapf

    mask, num = Ith_mask_mock(xs, ys, ms, verbose=False)

    stack_class = stacking_mock(inst)

    data = np.zeros([len(filt_order_arr), 4, 25])
    datasub = np.zeros([len(filt_order_arr), 4, 15])
    for ifilt, filt_order in enumerate(filt_order_arr):
        print('cat #%d, %d-th order filter'%(icat,filt_order))
        if filt_order==0:
            filtmap = srcmap - np.mean(srcmap[mask==1])
        else:   
            filtmap = image_poly_filter(srcmap, mask, degree=filt_order)

        for im, (m_min, m_max) in enumerate(zip(np.array(magbindict['m_min'])[mag_stack],
                                                np.array(magbindict['m_max'])[mag_stack])):
            print('cat #%d, %d-th order filter, stack %d < m < %d'\
                 %(icat,filt_order, m_min, m_max))
            spm = np.where((ms>=m_min) & (ms<m_max) &\
                           (xs>0) & (xs<1023) &\
                           (ys>0) & (ys<1023))[0]

            mapstack, maskstack = 0., 0.
            for i,(x, y, m, m_inband) in enumerate(zip(xs[spm],ys[spm],ms[spm],ms_inband[spm])):
                if i%100 == 0:
                    print('stack %d/%d sources'%(i, len(spm)))
                    
                make_srcmap_class.ms = np.array([m])
                make_srcmap_class.ms_inband = np.array([m_inband])
                make_srcmap_class.xls = np.array([x])
                make_srcmap_class.yls = np.array([y])
                srcmapi = make_srcmap_class.run_srcmap(ptsrc=True, verbose=False)

                maski, numi = Ith_mask_mock(np.array([x]), np.array([y]),
                                            np.array([m]), verbose=False)        

                filtmap_rm = filtmap - srcmapi

                num_rm = num - numi
                mask_rm = mask.copy()
                mask_rm[(maski==0) & (num==1)] = 1

                stack_class.xls = np.array([x])
                stack_class.yls = np.array([y])
                stack_class.ms = np.array([m])

                _, maskstacki, mapstacki =\
                 stack_class.run_stacking_bigpix(filtmap_rm, mask_rm, num_rm,
                                                           verbose=False,
                                                            return_profile=False)

                mapstack += mapstacki
                maskstack += maskstacki

            stack = np.zeros_like(mapstack)
            sp = np.where(maskstack!=0)
            stack[sp] = mapstack[sp] / maskstack[sp]
            stack[maskstack==0] = 0

            dx = stack_class.dx
            profile = radial_prof(np.ones([2*dx*10+1,2*dx*10+1]), dx*10, dx*10)
            rbinedges, rbins = profile['rbinedges'], profile['rbins']
            rsubbins, rsubbinedges = radial_binning(rbins, rbinedges)
            Nbins = len(rbins)
            Nsubbins = len(rsubbins)
            stackdat = {}
            stackdat['rbins'] = rbins*0.7
            stackdat['rbinedges'] = rbinedges*0.7 
            stackdat['rsubbins'] = rsubbins*0.7
            stackdat['rsubbinedges'] = rsubbinedges*0.7
 
            rbins /= 10 # bigpix
            rbinedges /=10 # bigpix
            radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx)
            prof_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
            for ibin in range(Nbins):
                spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                               (radmapstamp<rbinedges[ibin+1]))
                prof_arr[ibin] += np.sum(mapstack[spi])
                hit_arr[ibin] += np.sum(maskstack[spi])
            prof_norm = np.zeros_like(prof_arr)
            prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]

            stackdat['prof'] = prof_norm
            stackdat['profhit'] = hit_arr

            data[ifilt, im, :] = stackdat['prof']


            rsubbins /= 10 # bigpix
            rsubbinedges /=10 # bigpix
            prof_arr, hit_arr = np.zeros(Nsubbins), np.zeros(Nsubbins)
            for ibin in range(Nsubbins):
                spi = np.where((radmapstamp>=rsubbinedges[ibin]) &\
                               (radmapstamp<rsubbinedges[ibin+1]))
                prof_arr[ibin] += np.sum(mapstack[spi])
                hit_arr[ibin] += np.sum(maskstack[spi])
            prof_norm = np.zeros_like(prof_arr)
            prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]       
            stackdat['profsub'] = prof_norm
            stackdat['profhitsub'] = hit_arr

            datasub[ifilt, im, :] = stackdat['profsub']

    rbins = stackdat['rbins']
    rbinedges = stackdat['rbinedges']
    rsubbins = stackdat['rsubbins']
    rsubbinedges = stackdat['rsubbinedges']

    if save_data:
        data_dict = {'data': data, 'datasub': datasub, 
            'rbins':rbins, 'rbinedges':rbinedges, 
            'rsubbins':rsubbins, 'rsubbinedges':rsubbinedges,
            'filt_order_arr':filt_order_arr}
        fname  = savedir + 'filter_test_cen_TM%d_icat%d.pkl'%(inst, icat)

        with open(fname, "wb") as f:
            pickle.dump(data_dict , f)
    
    return data_dict

def run_micecat_auto_batch(inst ,ibatch, istart=0, batch_size=20, return_data=False, **kwargs):
    
    if return_data:
        data_dicts = [] 
    icat_arr = np.linspace(0, batch_size-1, batch_size) + ibatch*batch_size + istart
    icat_arr = icat_arr.astype(int)

    for icat in icat_arr:
        data_dict = run_micecat_auto_fliter_test_cen(inst, icat, **kwargs)

        if return_data:
            data_dicts.append(data_dict)

    if return_data:
        return data_dicts

    return

def get_micecat_sim_cen_auto(inst, im, sub=False, 
    filt_order=None, ratio=False, return_icat=False):
    '''
    Get the MICECAT central gal sim results.
    '''
    savedir='./micecat_data/all_fields/'
    typename = 'filter_test_cen'
    data_all = []
    icat_arr = []
    for icat in range(400):
        fname  = typename + '_TM%d_icat%d.pkl'%(inst, icat)
        if fname not in os.listdir(savedir):
            continue
        icat_arr.append(icat)
        
        with open(savedir + fname,"rb") as f:
            data_dict = pickle.load(f)
            
        if not sub:
            rbinedges = data_dict['rbinedges']
            rbins = data_dict['rbins']
            data = data_dict['data']
        else:
            rbinedges = data_dict['rsubbinedges']
            rbins = data_dict['rsubbins']
            data = data_dict['datasub']
            
        data_all.append(data)

    filt_order_arr = np.array(data_dict['filt_order_arr'])
    icat_arr = np.array(icat_arr)
    data_all = np.array(data_all)[...,im,:]
    
    if ratio:
        for i in range(data_all.shape[0]):
            nofilt = data_all[i,0,:].copy()
            for j in range(data_all.shape[1]):
                data_all[i,j,:] = data_all[i,j,:]/nofilt
                
    data_avg = np.mean(data_all, axis=0)
    data_std = np.std(data_all, axis=0)
    
    if filt_order is None:
        if return_icat:
            return rbins, data_avg, data_std, data_all, filt_order_arr, icat_arr
        return rbins, data_avg, data_std, data_all, filt_order_arr
    
    else:
        sp = np.where(filt_order_arr==filt_order)[0]
        data_avg = np.squeeze(data_avg[sp])
        data_std = np.squeeze(data_std[sp])
        data_all = np.squeeze(data_all[:,sp,:])
        if return_icat:
            return rbins, data_avg, data_std, data_all, icat_arr
        return rbins, data_avg, data_std, data_all

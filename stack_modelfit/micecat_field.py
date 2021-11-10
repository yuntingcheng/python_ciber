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
from astropy.coordinates import SkyCoord
from ciber_info import *
from cosmo_tools import *
from srcmap import *
from stack_ancillary import *

def run_IHL_Cl(ra_cent, dec_cent, abs_mag_cut=-18, m_th=20, bandname='ciber_I',
               logM_min=-np.inf, 
               z_min_arr = [0,0.2,0.4,0.6,0.8,1,1.2], 
               z_max_arr = [0.2,0.4,0.6,0.8,1,1.2,1.4],
               ihl_model = 'NFW',
               mask_IHL = False,
               verbose=True, savemaps=False):
    
    f_IHL_kwargs = {'logM_min':logM_min, 'f_IHL':1.}
    
    Cl_data = {'abs_mag_cut':abs_mag_cut, 'bandname': bandname, 'm_th':m_th,
               'ra':ra_cent, 'dec':dec_cent,
               'logM_min':f_IHL_kwargs['logM_min'],
               'z_min_arr':z_min_arr, 'z_max_arr':z_max_arr}
    
    print('ra = {}, dec = {}'.format(ra_cent, dec_cent))
    
    fname = mypaths['ciberdir']+'python_ciber/stack_modelfit/micecat_IHL_data/'\
    +'micecat_IHL_Cl_data_ra{}_dec{}_{}_mth{}.pkl'.format(ra_cent, dec_cent, ihl_model, m_th)
    
    if mask_IHL:
        fname = mypaths['ciberdir']+'python_ciber/stack_modelfit/micecat_IHL_data/'\
        +'micecat_IHL_Cl_data_ra{}_dec{}_{}_mth{}_maskIHL.pkl'\
        .format(ra_cent, dec_cent, ihl_model, m_th)

    mcfield = micecat_field(ra_cent, dec_cent,Nx=1024,Ny=1024)
    df = mcfield.get_micecat_df(add_fields=['sdss_r_abs_mag'])
    
    srcmap_all_tot = 0.
    srcmap_cen_tot = 0.
    srcmap_sat_tot = 0.
    srcmap_allcen_tot = 0.
    ihlmap_tot = 0.
    Cl_data['tot'] = {'z_min':z_min_arr[0], 'z_max':z_max_arr[-1]}
    for iz, (z_min, z_max) in enumerate(zip(z_min_arr, z_max_arr)):
        
        print('{} < z < {}'.format(z_min, z_max))

        Cl_data[iz] = {'z_min': z_min, 'z_max':z_max}

        dfi = df[(df.z_cgal >= z_min) & (df.z_cgal < z_max) \
                 & (df.sdss_r_abs_mag <= abs_mag_cut) \
                 & (df[bandname+'_true'] > m_th)]
        srcmap_all = mcfield.make_map(bandname, df=dfi)

        dfi = dfi[dfi.flag_central ==0]
        srcmap_cen = mcfield.make_map(bandname, df=dfi)
        srcmap_sat = srcmap_all - srcmap_cen

        dfi = df[(df.z_cgal >= z_min) & (df.z_cgal < z_max) \
                 & (df.sdss_r_abs_mag <= abs_mag_cut)]
        srcmap_allcen = mcfield.make_map_central(bandname,
                                                 df=dfi, band_mask=bandname, m_th=m_th)


        if mask_IHL:
            if ihl_model == 'uniform_disk':
                ihlmap = mcfield.make_ihlmap_uniform_disk(bandname, mcfield.f_IHL_const,
                                                          df=dfi, f_IHL_kwargs=f_IHL_kwargs,
                                                          band_mask=bandname, m_th=m_th,
                                                         verbose=verbose)
            elif ihl_model == 'NFW':            
                ihlmap = mcfield.make_ihlmap_NFW(bandname, mcfield.f_IHL_const,
                                                          df=dfi, f_IHL_kwargs=f_IHL_kwargs,
                                                          band_mask=bandname, m_th=m_th,
                                                         verbose=verbose)
        else:
            if ihl_model == 'uniform_disk':
                ihlmap = mcfield.make_ihlmap_uniform_disk(bandname, mcfield.f_IHL_const,
                                                          df=dfi, f_IHL_kwargs=f_IHL_kwargs,
                                                         verbose=verbose)
            elif ihl_model == 'NFW':            
                ihlmap = mcfield.make_ihlmap_NFW(bandname, mcfield.f_IHL_const,
                                                          df=dfi, f_IHL_kwargs=f_IHL_kwargs,
                                                         verbose=verbose)



        srcmap_all_tot += srcmap_all
        srcmap_cen_tot += srcmap_cen
        srcmap_sat_tot += srcmap_sat
        srcmap_allcen_tot += srcmap_allcen
        ihlmap_tot += ihlmap

        if savemaps:
            Cl_data[iz]['srcmap_cen'] = srcmap_cen
            Cl_data[iz]['srcmap_sat'] = srcmap_sat
            Cl_data[iz]['srcmap_allcen'] = srcmap_allcen
            Cl_data[iz]['ihlmap'] = ihlmap
            Cl_data['tot']['srcmap_cen'] = srcmap_cen_tot
            Cl_data['tot']['srcmap_sat'] = srcmap_sat_tot
            Cl_data['tot']['srcmap_allcen'] = srcmap_allcen_tot
            Cl_data['tot']['ihlmap'] = ihlmap_tot
        
        l,Cla, Claerr = get_power_spec(srcmap_all)
        l,Clc, Clcerr = get_power_spec(srcmap_cen)
        l,Cls, Clserr = get_power_spec(srcmap_sat)
        l,Clcs, Clcserr = get_power_spec(srcmap_cen, srcmap_sat)
        l,Cl2, Cl2err = get_power_spec(srcmap_allcen)

        l,Clh, Clherr = get_power_spec(ihlmap)
        l,Clha, Clhaerr = get_power_spec(ihlmap, srcmap_all)
        l,Clhc, Clhcerr = get_power_spec(ihlmap, srcmap_cen)
        l,Clhs, Clhserr = get_power_spec(ihlmap, srcmap_sat)
        l,Clh2, Clh2err = get_power_spec(ihlmap, srcmap_allcen)

        Cla_shsub = Cla-np.mean(Cla[-3:])
        Clc_shsub = Clc-np.mean(Clc[-3:])
        Cls_shsub = Cls-np.mean(Cls[-3:])
        Clcs_shsub = Clcs-np.mean(Clcs[-3:])
        Cl2_shsub = Cl2-np.mean(Cl2[-3:])
        Clh_shsub = Clh-np.mean(Clh[-3:])
        Clha_shsub = Clha-np.mean(Clha[-3:])
        Clhc_shsub = Clhc-np.mean(Clhc[-3:])
        Clhs_shsub = Clhs-np.mean(Clhs[-3:])
        Clh2_shsub = Clh2-np.mean(Clh2[-3:])
        
        Cl_data['l'] = l
        Cl_data[iz]['Cla'] = Cla
        Cl_data[iz]['Clc'] = Clc
        Cl_data[iz]['Cls'] = Cls
        Cl_data[iz]['Clcs'] = Clcs
        Cl_data[iz]['Cl2'] = Cl2
        Cl_data[iz]['Clh'] = Clh
        Cl_data[iz]['Clha'] = Clha
        Cl_data[iz]['Clhc'] = Clhc
        Cl_data[iz]['Clhs'] = Clhs
        Cl_data[iz]['Clh2'] = Clh2

        Cl_data[iz]['Cla_shsub'] = Cla_shsub
        Cl_data[iz]['Clc_shsub'] = Clc_shsub
        Cl_data[iz]['Cls_shsub'] = Cls_shsub
        Cl_data[iz]['Clcs_shsub'] = Clcs_shsub
        Cl_data[iz]['Cl2_shsub'] = Cl2_shsub
        Cl_data[iz]['Clh_shsub'] = Clh_shsub
        Cl_data[iz]['Clha_shsub'] = Clha_shsub
        Cl_data[iz]['Clhc_shsub'] = Clhc_shsub
        Cl_data[iz]['Clhs_shsub'] = Clhs_shsub
        Cl_data[iz]['Clh2_shsub'] = Clh2_shsub

        
        l,Cla, Claerr = get_power_spec(srcmap_all_tot)
        l,Clc, Clcerr = get_power_spec(srcmap_cen_tot)
        l,Cls, Clserr = get_power_spec(srcmap_sat_tot)
        l,Clcs, Clcserr = get_power_spec(srcmap_cen_tot, srcmap_sat_tot)
        l,Cl2, Cl2err = get_power_spec(srcmap_allcen_tot)

        l,Clh, Clherr = get_power_spec(ihlmap_tot)
        l,Clha, Clhaerr = get_power_spec(ihlmap_tot, srcmap_all_tot)
        l,Clhc, Clhcerr = get_power_spec(ihlmap_tot, srcmap_cen_tot)
        l,Clhs, Clhserr = get_power_spec(ihlmap_tot, srcmap_sat_tot)
        l,Clh2, Clh2err = get_power_spec(ihlmap_tot, srcmap_allcen_tot)

        Cla_shsub = Cla-np.mean(Cla[-3:])
        Clc_shsub = Clc-np.mean(Clc[-3:])
        Cls_shsub = Cls-np.mean(Cls[-3:])
        Clcs_shsub = Clcs-np.mean(Clcs[-3:])
        Cl2_shsub = Cl2-np.mean(Cl2[-3:])
        Clh_shsub = Clh-np.mean(Clh[-3:])
        Clha_shsub = Clha-np.mean(Clha[-3:])
        Clhc_shsub = Clhc-np.mean(Clhc[-3:])
        Clhs_shsub = Clhs-np.mean(Clhs[-3:])
        Clh2_shsub = Clh2-np.mean(Clh2[-3:])
        
        Cl_data['l'] = l
        Cl_data['tot']['Cla'] = Cla
        Cl_data['tot']['Clc'] = Clc
        Cl_data['tot']['Cls'] = Cls
        Cl_data['tot']['Clcs'] = Clcs
        Cl_data['tot']['Cl2'] = Cl2
        Cl_data['tot']['Clh'] = Clh
        Cl_data['tot']['Clha'] = Clha
        Cl_data['tot']['Clhc'] = Clhc
        Cl_data['tot']['Clhs'] = Clhs
        Cl_data['tot']['Clh2'] = Clh2

        Cl_data['tot']['Cla_shsub'] = Cla_shsub
        Cl_data['tot']['Clc_shsub'] = Clc_shsub
        Cl_data['tot']['Cls_shsub'] = Cls_shsub
        Cl_data['tot']['Clcs_shsub'] = Clcs_shsub
        Cl_data['tot']['Cl2_shsub'] = Cl2_shsub
        Cl_data['tot']['Clh_shsub'] = Clh_shsub
        Cl_data['tot']['Clha_shsub'] = Clha_shsub
        Cl_data['tot']['Clhc_shsub'] = Clhc_shsub
        Cl_data['tot']['Clhs_shsub'] = Clhs_shsub
        Cl_data['tot']['Clh2_shsub'] = Clh2_shsub

    with open(fname, "wb") as f:
        pickle.dump(Cl_data , f)

    return Cl_data

class micecat_field:
    
    def __init__(self, ra_cent, dec_cent, pix_size=7, Nx=1024, Ny=1024):
        '''
        retrive a rectangular MICECAT field with size
        ra_size[deg] x dec_size[deg] along ra and dec axis
        centered at ra_cent[deg], deg_cent[deg]
        
        make a ra_size[deg] x dec_size[deg] sized map
        with pix_size[arcsec]
        '''
        self.ra_cent = ra_cent
        self.dec_cent = dec_cent
        self.pix_size = pix_size
        self.Nx = Nx
        self.Ny = Ny
        self.ra_size = pix_size * Nx / 3600
        self.dec_size = pix_size * Ny / 3600
        
        return
    
    def get_micecat_df(self, add_fields=None, add_Rvir=True):
        
        df = self.get_raw_df()
        
        fields = [
        'unique_gal_id', 'unique_halo_id', 'flag_central', 'nsats',
        'ra_gal', 'dec_gal', 'ra1', 'dec1', 'x', 'y',
        'z_cgal', 'lmhalo', 'lsfr', 'lmstellar',
        'ciber_I_true', 'ciber_H_true']
        
        if add_fields is not None:
            for fieldname in add_fields:
                if fieldname not in fields:
                    fields.append(fieldname)

        df['ciber_I_true'] = df['euclid_nisp_y_true']
        Hmag = 0.5*(10**(-df['euclid_nisp_j_true']/2.5) \
                    + 10**(-df['euclid_nisp_h_true']/2.5))
        Hmag = -2.5*np.log10(Hmag)
        df['ciber_H_true'] = Hmag

        ra1, dec1 = self.coord_transform(df.ra_gal.values, df.dec_gal.values)
        df['ra1'], df['dec1'] = ra1, dec1

        Nx_mid, Ny_mid = (self.Nx-1)/2, (self.Ny-1)/2
        df['x'] = (df.ra1)/(np.cos(df.dec1*np.pi/180))*3600/self.pix_size + Nx_mid
        df['y'] = (df.dec1)*3600/self.pix_size + Ny_mid

        if add_Rvir:
            z_arr = np.array(df['z_cgal'])
            Mh_arr = 10**np.array(df['lmhalo'])
            rhoc_arr = np.array(cosmo.critical_density(z_arr).to(u.M_sun / u.Mpc**3))
            rvir_arr = ((3 * Mh_arr) / (4 * np.pi * 200 * rhoc_arr))**(1./3)
            DA_arr = np.array(cosmo.comoving_distance(z_arr))
            rvir_ang_arr = (rvir_arr / DA_arr) * u.rad.to(u.arcsec)
            df['Rv_Mpc'] = rvir_arr
            df['Rv_arcsec'] = rvir_ang_arr
            
            fields.append('Rv_Mpc')
            fields.append('Rv_arcsec')

        df = df[fields] 
        
        return df
    
    def get_raw_df(self):
        
        fname = mypaths['MCcatdat'] + 'all_fields/' \
                        + 'micecat_ra%d_dec%d_%.1fx%.1f.csv.bz2'\
                        %(self.ra_cent,self.dec_cent,self.ra_size, self.dec_size)

        if not os.path.exists(fname):
            self.run_micecat_query()
        
        with bz2.BZ2File(fname) as catalog_fd:
            df = pd.read_csv(catalog_fd, sep=",", index_col=False,
                             comment='#', na_values=r'\N')

        for name in list(df):
            if 'true' in name:
                df[name] = df[name].values \
                - 0.8 * (np.arctan(1.5 * df['z_cgal'].values) - 0.1489)
            
        return df

    def make_map(self, bandname, df=None, 
        x_arr=None, y_arr=None, m_arr=None):

        if (x_arr is None) or (y_arr is None) or (m_arr is None):
            if df is None:
                df = self.get_micecat_df(add_fields=[bandname + '_true'])
        
        m_arr = df[bandname + '_true'].values if m_arr is None else m_arr
        x_arr = df.x.values if x_arr is None else x_arr
        y_arr = df.y.values if y_arr is None else y_arr

        sp = np.where(m_arr!=-999.)[0]
        x_arr = x_arr[sp]
        y_arr = y_arr[sp]
        m_arr = m_arr[sp]
        wl = self.filter_wleff(bandname)
        sr = ((self.pix_size/3600.0)*(np.pi/180.0))**2
        I = 3631. * 10**(-m_arr / 2.5) * (3 / wl) * 1e6 / (sr*1e9)
        srcmap = np.histogram2d(x_arr, y_arr,
                                [np.arange(self.Nx+1)-0.5,
                                np.arange(self.Ny+1)-0.5],
                                weights=I)[0]
        return srcmap


    def make_map_central(self, bandname, df=None, 
                         idx_mask=None, band_mask=None, m_th=None):
        
        if df is None:
            df = self.get_micecat_df(add_fields=[bandname + '_true'])
        else:
            df = df.copy()
        
        dfc = self.dfcentral_from_df(bandname, df, idx_mask=idx_mask,
                                     band_mask=band_mask, m_th=m_th)

        srcmap = self.make_map(bandname, df=dfc,
                               m_arr=dfc[bandname + '_true_sum'].values)
        
        return srcmap

    def dfcentral_from_df(self, bandname, df=None, 
                          idx_mask=None, band_mask=None, m_th=None):
        if df is None:
            df = self.get_micecat_df(add_fields=[bandname + '_true'])
        else:
            df = df.copy()
        
        if idx_mask is None:
            if m_th is None:
                idx_mask = np.array([])
            else:
                band_mask = bandname if band_mask is None else bandname
                idx_mask = np.where(df[band_mask + '_true'].values < m_th)[0]
        else:
            idx_mask = np.array(idx_mask)
        
        Fnu = 3631 * 10**(-df[bandname + '_true'].values / 2.5)
        if len(idx_mask) > 0:
            Fnu[idx_mask] = 0.
        df['Fnu'] = Fnu
        dfc = df[df.flag_central==0].copy()
        dfsum = df.groupby('unique_halo_id',as_index=False)[['Fnu']].sum()
        dfsum.rename(columns={'Fnu':'Fnu_sum'}, inplace=True)
        dfc = dfc.merge(dfsum, on='unique_halo_id')
        
        Fnu_sum = dfc['Fnu_sum'].values
        mag_sum = np.ones(len(dfc)) * -999.
        mag_sum[Fnu_sum!=0] = -2.5 * np.log10(Fnu_sum[Fnu_sum!=0]/3631)
        dfc[bandname + '_true_sum'] = mag_sum
        
        return dfc

    def make_data_cube(self, z_mid, bandname=None, df=None,
                       cube_size=None, vox_size=1):
        if df is None:
            df = self.get_micecat_df(add_fields=[bandname + '_true'])
        else:
            df = df.copy()

        bandname = 'Mhalo' if bandname is None else bandname
        
        chi_mid = cosmo.comoving_distance(z_mid).value
        if cube_size is None:
            z_arr = np.arange(0, z_mid, 0.00001)
            chi_arr = cosmo.comoving_distance(z_arr).value
            width_arr = chi_arr*(self.ra_size * u.deg).to(u.rad).value
            z_idx = np.where((chi_mid - chi_arr)*2 < width_arr)[0][0]
            cube_size = (chi_mid - chi_arr[z_idx])*2
        Nside = int(cube_size//vox_size)
        cube_size = Nside * vox_size

        chi_arr = cosmo.comoving_distance(df.z_cgal.values).value
        idx = np.where(np.abs(chi_arr-chi_mid) < cube_size/2)[0]
        dfi = df.iloc[idx]
        chi_arr = chi_arr[idx]
        ra_arr, dec_arr = self.coord_transform(dfi.ra_gal.values, dfi.dec_gal.values)
        x_arr = ra_arr * u.deg.to(u.rad) * chi_arr
        y_arr = dec_arr * u.deg.to(u.rad) * chi_arr
        z_arr = chi_arr - chi_mid

        if bandname == 'Mhalo':
            weights_arr = 10**dfi.lmhalo.values
        else:
            m_arr = dfi[bandname + '_true'].values
            Fnu_arr = 3631 * 10**(-m_arr/2.5) * u.Jy
            DL_arr = cosmo.luminosity_distance(dfi.z_cgal.values)
            Lnu_arr = (Fnu_arr*4*np.pi*DL_arr**2).to(u.Lsun/u.Hz)
            weights_arr = Lnu_arr.value

        binedges = np.arange(0, vox_size*(Nside+0.1), vox_size)
        binedges -= (binedges[0] + binedges[-1]) / 2
        cube,_ = np.histogramdd(np.stack((x_arr,y_arr,z_arr),axis=1),
                             bins = (binedges, binedges, binedges),
                             weights = weights_arr)
        
        return cube
    

    def _rebin_map_coarse(self, original_map, Nsub):
        m, n = np.array(original_map.shape)//(Nsub, Nsub)
        return original_map.reshape(m, Nsub, n, Nsub).mean((1,3))


    def f_IHL_const(self, logMh_arr, f_IHL=0.1, logM_min=12):
        '''
        a constant IHL frac for Mh > 10**lnM_min
        '''

        f_IHL_arr = np.ones_like(logMh_arr) * f_IHL
        f_IHL_arr[logMh_arr < logM_min] = 0 

        return f_IHL_arr

    def make_ihlmap_uniform_disk(self, bandname, f_IHL_func,
                                 df=None, f_IHL_kwargs={},
                                 band_mask=None, m_th=None,
                                 Nsub=2, verbose=True):
        if df is None:
            df = self.get_micecat_df(add_fields=[bandname + '_true'])
        else:
            df = df.copy()

        dx = 10 * Nsub

        dfc = self.dfcentral_from_df(bandname, df, band_mask=band_mask, m_th=m_th)
        dfc['f_IHL'] = f_IHL_func(dfc.lmhalo.values, **f_IHL_kwargs)
        dfc = dfc[dfc.f_IHL > 0]
        dfc['Fnu_IHL'] = dfc.Fnu_sum * dfc.f_IHL

        wl = self.filter_wleff(bandname)
        sr = ((self.pix_size/3600.0)*(np.pi/180.0))**2
        Is = dfc.Fnu_IHL.values * (3 / wl) * 1e6 / (sr*1e9)
        ihlmap_large = np.zeros((self.Nx*Nsub + 4*dx, self.Ny*Nsub + 4*dx))

        Rs = dfc.Rv_arcsec.values / (self.pix_size/Nsub)
        xss = np.round(dfc.x.values * Nsub + (Nsub/2 - 0.5) + 2 * dx).astype(np.int32)
        yss = np.round(dfc.y.values * Nsub + (Nsub/2 - 0.5) + 2 * dx).astype(np.int32)
        xx, yy = np.meshgrid(np.arange(ihlmap_large.shape[0]),
                             np.arange(ihlmap_large.shape[1]), indexing='ij')

        for i,(xs,ys,I,R) in enumerate(zip(xss, yss, Is, Rs)):
            radmap = (xx - xs)**2 + (yy - ys)**2
            sp_in = np.where(radmap <= R**2)
            if len(sp_in[0]) > 0:
                ihlmap_large[sp_in] += I / len(sp_in[0])

            if len(Is)>20:
                if verbose and i%(len(Is)//20)==0:
                    print('run IHL map %d / %d (%.1f %%)'\
                          %(i, len(Is), i/len(Is)*100))

        ihlmap = self._rebin_map_coarse(ihlmap_large, Nsub)*Nsub**2
        ihlmap = ihlmap[2*dx//Nsub : 2*dx//Nsub+self.Nx,\
                        2*dx//Nsub : 2*dx//Nsub+self.Ny]

        return ihlmap
        
    def make_ihlmap_NFW(self, bandname, f_IHL_func, df=None, Rvir_lim=1.,
                        band_mask=None, m_th=None,
                        f_IHL_kwargs={}, Nsub=2, verbose=True):
        if df is None:
            df = self.get_micecat_df(add_fields=[bandname + '_true'])
        else:
            df = df.copy()

        dx = 10 * Nsub

        dfc = self.dfcentral_from_df(bandname, df, band_mask=band_mask, m_th=m_th)
        dfc['f_IHL'] = f_IHL_func(dfc.lmhalo.values, **f_IHL_kwargs)
        dfc = dfc[dfc.f_IHL > 0]
        dfc['Fnu_IHL'] = dfc.Fnu_sum * dfc.f_IHL

        wl = self.filter_wleff(bandname)
        sr = ((self.pix_size/3600.0)*(np.pi/180.0))**2
        Is = dfc.Fnu_IHL.values * (3 / wl) * 1e6 / (sr*1e9)
        ihlmap_large = np.zeros((self.Nx*Nsub + 4*dx, self.Ny*Nsub + 4*dx))

        Rs = dfc.Rv_arcsec.values / (self.pix_size/Nsub)
        xss = np.round(dfc.x.values * Nsub + (Nsub/2 - 0.5) + 2 * dx).astype(np.int32)
        yss = np.round(dfc.y.values * Nsub + (Nsub/2 - 0.5) + 2 * dx).astype(np.int32)
        xx, yy = np.meshgrid(np.arange(ihlmap_large.shape[0]),
                             np.arange(ihlmap_large.shape[1]), indexing='ij')
        zs = dfc.z_cgal.values
        Mhs = 10**dfc.lmhalo.values * cosmo.h
        
        NFW = NFW_proile()
        for i,(xs,ys,I,R,z,Mh) in enumerate(zip(xss, yss, Is, Rs, zs, Mhs)):
            radmap = (xx - xs)**2 + (yy - ys)**2
            sp_in = np.where(radmap <= (R*Rvir_lim)**2)
            if len(sp_in[0]) > 0:
                ihlmapi = NFW.NFW_2d(np.sqrt(radmap[sp_in])*self.pix_size/Nsub, z, Mh)
                if np.sum(ihlmapi)==0:
                    ihlmapi[:] = I
                ihlmap_large[sp_in] += (I * ihlmapi / np.sum(ihlmapi))
            if len(Is)>20:
                if verbose and i%(len(Is)//20)==0:
                    print('run IHL map %d / %d (%.1f %%)'\
                          %(i, len(Is), i/len(Is)*100))

        ihlmap = self._rebin_map_coarse(ihlmap_large, Nsub)*Nsub**2
        ihlmap = ihlmap[2*dx//Nsub : 2*dx//Nsub+self.Nx,\
                        2*dx//Nsub : 2*dx//Nsub+self.Ny]

        return ihlmap
           
    def filter_wleff(self, name):
        '''
        get effective wavelength [um] for a given filter

        cosmos: 
        https://cosmos.astro.caltech.edu/page/filterset
        DES:
        http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=CTIO&gname2=DECam&asttype=
        euclid:
        http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Euclid&asttype=
        vhs:
        http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Paranal&gname2=VISTA&asttype=
        sdss:
        http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=SLOAN&asttype=
        '''
        bandwl_dict = {'sdss_u':0.360804, 'sdss_g':0.467178, 'sdss_r':0.614112, 
                       'sdss_i':0.745789, 'sdss_z':0.892278,
                       'vhs_j':1.248100, 'vhs_h':1.634819, 'vhs_ks':2.143546,
                       'euclid_vis_riz':0.687271, 'euclid_nisp_y':1.073286,
                       'euclid_nisp_j':1.343364, 'euclid_nisp_h':1.744004,
                       'des_asahi_full_g':0.477084, 'des_asahi_full_r':0.637133, 
                       'des_asahi_full_i':0.777419, 'des_asahi_full_z':0.915790,
                       'des_asahi_full_y':0.988635,
                       'cosmos_cfht_u':0.382329, 'cosmos_cfht_i':0.761766,
                       'cosmos_wfcam_j':1.249102, 'cosmos_wircan_ks':2.159044,
                       'cosmos_subaru_b':0.445832,'cosmos_subaru_v':0.547783,
                       'cosmos_subaru_g':0.477707,'cosmos_subaru_r':0.628871,
                       'cosmos_subaru_i':0.768388,'cosmos_subaru_z':0.903688,
                       'ciber_I':1.05, 'ciber_H':1.79
                      }

        return bandwl_dict[name]

    def coord_transform(self, ra, dec, ra_cent=None, dec_cent=None):
        '''
        Get transformed RA, Dec if shift the ra dec origin to ra_cent, dec_cent
        '''
        
        ra_cent = self.ra_cent if ra_cent is None else ra_cent
        dec_cent = self.dec_cent if dec_cent is None else dec_cent
        
        s_ra_d = np.sin((ra - ra_cent) * np.pi/180)
        c_ra_d = np.cos((ra - ra_cent) * np.pi/180)
        s_ra_c, c_ra_c = np.sin((ra_cent) * np.pi/180), np.cos((ra_cent) * np.pi/180)
        s_dec_c, c_dec_c = np.sin((dec_cent) * np.pi/180), np.cos((dec_cent) * np.pi/180)
        s_ra_x, c_ra_x = np.sin((ra) * np.pi/180), np.cos((ra) * np.pi/180)
        s_dec_x, c_dec_x = np.sin((dec) * np.pi/180), np.cos((dec) * np.pi/180)

        tan_ra1 = s_ra_d / (c_dec_c * c_ra_d + s_dec_c * s_dec_x / c_dec_x)
        sin_dec1 = -c_dec_x * s_dec_c * c_ra_d + c_dec_c * s_dec_x

        dec1 = np.arctan(tan_ra1) * 180 / np.pi
        ra1 = np.arcsin(sin_dec1) * 180 / np.pi
        
        return ra1, dec1
    
    def get_corner_coord(self, scale=1):
        
        scale1 = scale * 1.2
        dec_max = self.dec_cent + self.dec_size*scale1/2
        dec_min = self.dec_cent - self.dec_size*scale1/2
        ra_max1 = self.ra_cent + self.ra_size*scale1/2 / np.cos(dec_max * np.pi / 180)
        ra_max2 = self.ra_cent - self.ra_size*scale1/2 / np.cos(dec_max * np.pi / 180)
        ra_min1 = self.ra_cent + self.ra_size*scale1/2 / np.cos(dec_min * np.pi / 180)
        ra_min2 = self.ra_cent - self.ra_size*scale1/2 / np.cos(dec_min * np.pi / 180)
        
        ra_max = np.max((ra_max1, ra_max2, ra_min1, ra_min2))
        ra_min = np.min((ra_max1, ra_max2, ra_min1, ra_min2))
        ra_arr = np.linspace(ra_min, ra_max,1000)
        dec_arr = np.linspace(dec_min, dec_max,1000)
        dec_grid, ra_grid = np.meshgrid(dec_arr, ra_arr)
        ra1_grid, dec1_grid = self.coord_transform(ra_grid, dec_grid)
        c = SkyCoord(ra1_grid*u.deg, dec1_grid*u.deg, frame='icrs')
        
        ra_corner, dec_corner = [], []
        for sign in ((1,1), (1,-1), (-1,1), (-1,-1)):
            c0 = SkyCoord(sign[0]*scale*self.ra_size/2*u.deg,
                          sign[1]*scale*self.dec_size/2*u.deg, frame='icrs')
            sep = c.separation(c0).deg
            ra_corner.append(ra_grid[np.where(sep==np.min(sep))][0])
            dec_corner.append(dec_grid[np.where(sep==np.min(sep))][0])

        return np.array(ra_corner), np.array(dec_corner)
    
    def get_corner_coord_transformed(self, scale=1):
        
        ra_corner, dec_corner = self.get_corner_coord(scale=scale)
        ra1_corner, dec1_corner = self.coord_transform(ra_corner, dec_corner)
        
        return ra1_corner, dec1_corner
    
    def cat_download_coord_limit(self, coords=None, scale=1.2):
        
        if coords is not None:
            return coords
        
        ra_corner, dec_corner = self.get_corner_coord(scale=scale)
        ra_min, ra_max = np.min(ra_corner), np.max(ra_corner)
        dec_min, dec_max = np.min(dec_corner), np.max(dec_corner)
        
        return (ra_min, ra_max, dec_min, dec_max)
    
    def gen_sql(self, select_fields=None, select_top=None, coords=None, scale=1.2):
        
        ra_min, ra_max, dec_min, dec_max \
        = self.cat_download_coord_limit(coords=coords, scale=scale)
        if select_fields is None:
            select_fields = [
            'unique_gal_id', 'unique_halo_id', 'flag_central', 'nsats', 'ra_gal', 'dec_gal',
            'z_cgal', 'lmhalo', 'lsfr', 'lmstellar', 'sed_cos', 'sed_peg',
            'des_asahi_full_g_true', 'des_asahi_full_r_true',
            'des_asahi_full_i_true', 'des_asahi_full_z_true', 'des_asahi_full_y_true',
            'euclid_vis_riz_true', 'euclid_nisp_y_true', 
            'euclid_nisp_j_true', 'euclid_nisp_h_true',
            'cosmos_cfht_u_true', 'cosmos_subaru_b_true', 
            'cosmos_subaru_v_true', 'cosmos_subaru_g_true',
            'cosmos_subaru_r_true', 'cosmos_subaru_i_true', 'cosmos_subaru_z_true',
            'cosmos_cfht_i_true', 'cosmos_wfcam_j_true', 'cosmos_wircan_ks_true', 
            'sdss_u_true', 'sdss_g_true', 'sdss_r_true', 'sdss_i_true', 'sdss_z_true', 
            'sdss_r_abs_mag', 'vhs_j_true', 'vhs_h_true','vhs_ks_true',
            ]

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
    
    def query_micecat(self):
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
                driver.find_element_by_xpath("//button[@ng-click='query.readonly = !query.readonly;"\
                                             +"query.showQueryHelp = true']").click()
                break
            except NoSuchElementException:
                time.sleep(wait_time)

        return driver


    def sendSQL_micecat(self, driver, sql):
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
                request_id \
                = driver.find_element_by_xpath("//text[@ng-bind-html='request.message']").text

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
    
    def dowload_micecat(self, request_id):
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

        while True:
            try:
                driver.find_element_by_xpath("//a[contains(@href, %s)]"%(str(request_id))).click()
                break
            except NoSuchElementException:
                time.sleep(wait_time)

        return driver

    def process_downloaded_micecat(self,request_id):
                
        fname = 'micecat_ra%d_dec%d_%.1fx%.1f.csv.bz2'\
                        %(self.ra_cent,self.dec_cent,self.ra_size, self.dec_size)
        try:
            os.rename('/Users/ytcheng/Downloads/' + str(request_id) + '.csv.bz2',
                     '/Users/ytcheng/Downloads/' + fname)
            shutil.move('/Users/ytcheng/Downloads/' + fname,
                        mypaths['MCcatdat'] + 'all_fields/' + fname)

        except FileNotFoundError:
            return
        return

    def run_micecat_query(self, sql_kwargs={}):
        
        driver = self.query_micecat()
        sql = self.gen_sql(**sql_kwargs)
        request_id = self.sendSQL_micecat(driver, sql)
        print('query MICECAT ra = %d dec = %d, request id %d'\
              %(self.ra_cent, self.dec_cent, request_id))
        driver.quit()
        
        time.sleep(120)
        driver = self.dowload_micecat(request_id)
        time.sleep(60)
        driver.quit()
        self.process_downloaded_micecat(request_id)
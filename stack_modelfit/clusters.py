from utils import *
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord

class clusters:
    def __init__(self, inst, ifield, zrange=(0, np.inf),
                 lnMhrange=(0, np.inf), abell_rad=500):
        self.inst = inst
        self.ifield = ifield
        self.field = fieldnamedict[ifield]
        self.zrange = zrange
        self.lnMhrange = lnMhrange
        self.abell_rad = abell_rad
    
    def cluster_mask(self):
        inst = self.inst
        dfa = self.abell_src()
        dfc = self.sdss_src()
        
        mask = np.ones([1024, 1024])
        
        for x,y in zip(dfa['x'+str(inst)], dfa['y'+str(inst)]):
            radmap = make_radius_map(mask, x, y)*7
            mask[radmap<self.abell_rad] = 0
        
        for x, y, r in zip(dfc['x'+str(inst)], dfc['y'+str(inst)], dfc['r200_arcsec']):
            radmap = make_radius_map(mask, x, y)*7
            mask[radmap<r] = 0
            
        return mask
            
    def sdss_src(self):
        datadir = mypaths['ciberdir'] + 'doc/20170617_Stacking/maps/clustercats/'
        dfc = pd.read_csv(datadir + self.field + '.csv')
        dfc = self._cat_add_xy(self.field, dfc)
        dfc = dfc[(dfc['x1']>-0.5) & (dfc['x1']<1023.5) & (dfc['y1']>-0.5) & (dfc['y1']<1023.5)]
        dfc['r200_arcsec'] = np.array(cosmo.arcsec_per_kpc_proper(dfc['zph'])) * dfc['r200'].values * 1e3
        rhoc_arr = np.array(cosmo.critical_density(dfc['zph'].values).to(u.M_sun / u.Mpc**3))
        Mh_arr = (4/3*np.pi*dfc['r200']**3)*200*rhoc_arr
        dfc['lmhalo'] = np.log10(Mh_arr)
        dfc = dfc[(dfc['zph']>self.zrange[0]) & (dfc['zph']<self.zrange[1]) &\
                 (dfc['lmhalo']>self.lnMhrange[0]) & (dfc['lmhalo']<self.lnMhrange[1])]
        
        return dfc
    
    def abell_cat(self):
        datadir = mypaths['ciberdir'] + 'doc/20170617_Stacking/maps/clustercats/'
        df = pd.read_csv(datadir + 'abell_clusters.txt',
                         skiprows=2, sep='|')
        df.drop(['Unnamed: 0', 'Unnamed: 10'],axis=1, inplace=True)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        ra_arr = []
        dec_arr = []
        for sra, sdec in zip(df['ra'], df['dec']):
            ra_hr, ra_min = sra.split()
            ra = (np.float(ra_hr) + np.float(ra_min) / 60) * 15
            dec_int, dec_dec = sdec.split()
            dec = np.float(dec_int) +  np.sign(np.float(dec_int)) * np.float(dec_dec) / 100
            ra_arr.append(ra)
            dec_arr.append(dec)
        df['ra'] = ra_arr
        df['dec'] = dec_arr
        
        return df
    
    def abell_src(self):
        df = self.abell_cat()
        ra_cent, dec_cent = field_center_dict[self.ifield]
        dfa = df[(df['ra']>ra_cent-2) & (df['ra']<ra_cent+2) & \
                 (df['dec']>dec_cent-2) & (df['dec']<dec_cent+2)].copy()
        dfa = self._cat_add_xy(self.field, dfa)
        dfa = dfa[(dfa['x1']>0) & (dfa['x1']<1024) & \
                 (dfa['x2']>0) & (dfa['x2']<1024)].copy()
        dfa.drop(['bmtype','rich','dist'],axis=1,inplace=True)
        return dfa

    def _cat_add_xy(self,field, df):
        if len(df)==0:
            order = [c for c in df.columns]
            df['x1'] = []
            df['x2'] = []
            df['y1'] = []
            df['y2'] = []
            order = order[:3] + ['x1','y1','x2','y2'] + order[3:]
            dfout = df[order].copy()
            return df

        order = [c for c in df.columns]
        # find the x, y solution with all quad
        for inst in [1,2]:
            hdrdir = mypaths['ciberdir'] + 'doc/20170617_Stacking/maps/astroutputs/inst' + str(inst) + '/'
            xoff = [0,0,512,512]
            yoff = [0,512,0,512]
            for iquad,quad in enumerate(['A','B','C','D']):
                hdulist = fits.open(hdrdir + field + '_' + quad + '_astr.fits')
                wcs_hdr=wcs.WCS(hdulist[('primary',1)].header, hdulist)
                hdulist.close()
                src_coord = SkyCoord(ra=df['ra']*u.degree, dec=df['dec']*u.degree, frame='icrs')

                x_arr, y_arr = wcs_hdr.all_world2pix(df['ra'],df['dec'],0)
                df['x' + quad] = x_arr + xoff[iquad]
                df['y' + quad] = y_arr + yoff[iquad]

            df['meanx'] = (df['xA'] + df['xB'] + df['xC'] + df['xD']) / 4
            df['meany'] = (df['yA'] + df['yB'] + df['yC'] + df['yD']) / 4

            # assign the x, y with the nearest quad solution
            df['x'+str(inst)] = df['xA'].copy()
            df['y'+str(inst)] = df['yA'].copy()
            bound = 511.5
            df.loc[ (df['meanx'] < bound) & (df['meany'] > bound),'x'+str(inst)] = df['xB']
            df.loc[ (df['meanx'] < bound) & (df['meany'] > bound),'y'+str(inst)] = df['yB']

            df.loc[ (df['meanx'] > bound) & (df['meany'] < bound),'x'+str(inst)] = df['xC']
            df.loc[ (df['meanx'] > bound) & (df['meany'] < bound),'y'+str(inst)] = df['yC']

            df.loc[ (df['meanx'] > bound) & (df['meany'] > bound),'x'+str(inst)] = df['xD']
            df.loc[ (df['meanx'] > bound) & (df['meany'] > bound),'y'+str(inst)] = df['yD']

        # write x, y to df
        order = order[:3] + ['x1','y1','x2','y2'] + order[3:]
        dfout = df[order].copy()

        return dfout
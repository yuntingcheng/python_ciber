from stack_ancillary import *

class stacking:
    def __init__(self, inst, ifield, m_min, m_max, srctype='g', 
        savename=None, load_from_file=False, loaddir=None, filt_order=2,
         run_nonuniform_BG=False, getBG=True, BGsub=True, all_src=False,
         subsub=False, uniform_jack=False, savemaps=False):
        self.inst = inst
        self.ifield = ifield
        self.field = fieldnamedict[ifield]
        self.m_min = m_min
        self.m_max = m_max
        self.filt_order = filt_order
        self.getBG = getBG
        self.BGsub = BGsub
        self.subsub = subsub
        self.uniform_jack = uniform_jack
        self.savemaps = savemaps
        self.data_maps = None

        if loaddir is None:
            loaddir = './stack_data/'
        if savename is None:
            savename = loaddir + 'stackdat_TM%d_%s_%d_%d_filt%d'\
            %(inst, self.field, m_min, m_max, filt_order)
            if uniform_jack:
                savename = loaddir + 'stackdat_TM%d_%s_%d_%d_filt%d_unijack'\
                %(inst, self.field, m_min, m_max, filt_order)

        if all_src:
            # This has no z & Mh cut
            savename = loaddir + 'stackdat_TM%d_%s_%d_%d'\
            %(inst, self.field, m_min, m_max)
            if uniform_jack:
                savename = loaddir + 'stackdat_TM%d_%s_%d_%d_unijack'\
                %(inst, self.field, m_min, m_max)


        self.savename = savename
        
        if load_from_file:
            stackdat = np.load(savename + '.npy' ,allow_pickle='TRUE').item()
            self.stackdat = stackdat
            if run_nonuniform_BG:
                self.stack_BG(Nbg=64, uniform=uniform_jack)
                np.save(savename, stackdat)

        else:
            if uniform_jack:
                stackdat = self.stack_PS(sample_type='jack_random')
            else:
                stackdat = self.stack_PS(sample_type='jack_region')
            self.stackdat = stackdat
            self.stack_BG(uniform=uniform_jack)
            np.save(savename, stackdat)
        
        self._post_process()
    
    def _post_process(self):
        if self.subsub:
            self._get_subsubbins(Nrebin=6)
        self._get_jackknife_profile()
        self._get_covariance()
        if self.getBG:
            self._get_BG_jackknife_profile()
            self._get_BG_covariance()
        self._get_BGsub()
        self._get_PSF_from_data()
        self._get_PSF_covariance_from_data()
        if self.subsub:
            self._get_PSF_subsubbins(Nrebin=6)
        self._get_ex_covariance()
        self._get_excess()
        
    def stack_PS(self, srctype='g', dx=1200, 
        sample_type='jack_region', unmask=True, verbose=True,
        cliplim=None, srcdat=None):

        inst = self.inst
        ifield = self.ifield
        m_min, m_max = self.m_min, self.m_max

        if self.data_maps is None:
            data_maps = {1: image_reduction(1), 2: image_reduction(2)}
            self.data_maps = data_maps
        else:
            data_maps = self.data_maps

        cbmap, psmap, strmask, strnum, mask_inst1, mask_inst2 = \
        load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                            (inst,ifield,'psmap'),
                                           (inst,ifield,'strmask'), 
                                           (inst,ifield,'strnum'),
                                           (1,ifield,'mask_inst'),
                                           (2,ifield,'mask_inst')])
        if inst==1:
            mask_inst = mask_inst1
        else:
            mask_inst = mask_inst2
        
        cbmap = image_poly_filter(cbmap, strmask*mask_inst, degree=self.filt_order)
        
        if srcdat is None:
            srcdat = ps_src_select(inst, ifield, m_min, m_max, 
                [mask_inst1, mask_inst2], sample_type=sample_type)

            if srcdat['N' + srctype] < 64:
                srcdat = ps_src_select(inst, ifield, m_min, m_max, 
                    [mask_inst1, mask_inst2], sample_type='jack_random',
                    Nsub=srcdat['N' + srctype])                
        if cliplim is None:
            cliplim = self._stackihl_PS_cliplim()

        # init stackdat
        stackdat = {}
        stackdat['rbins'] = cliplim['rbins']
        stackdat['rbinedges'] = cliplim['rbinedges']
        stackdat['rsubbins'],stackdat['rsubbinedges'] =\
        self._radial_binning(cliplim['rbins'], cliplim['rbinedges'])
        stackdat['inst']= inst
        stackdat['ifield'] = ifield
        stackdat['field'] = fieldnamedict[ifield]
        stackdat['m_min'], stackdat['m_max'] = m_min, m_max
        stackdat['Nsrc'] = srcdat['N' + srctype]
        stackdat['Nsub'] = srcdat['Nsub']
        stackdat['sub'] = {}
        
        # start stacking
        Nbins = len(stackdat['rbins'])
        Nsubbins = len(stackdat['rsubbins'])
        radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx) # subpix unit
        rbinedges = stackdat['rbinedges']/0.7 # subpix unit
        rsubbinedges = stackdat['rsubbinedges']/0.7 # subpix unit
        
        cbmapstack, psmapstack, maskstack =\
         np.zeros([2*dx+1, 2*dx+1]),np.zeros([2*dx+1, 2*dx+1]), np.zeros([2*dx+1, 2*dx+1])
        start_time = time.time()
        for isub in range(srcdat['Nsub']):
            stackdat['sub'][isub] = {}

            xls = srcdat['sub'][isub]['x' + srctype + '_arr']
            yls = srcdat['sub'][isub]['y' + srctype + '_arr']
            
            stackdat['sub'][isub]['Nsrc'] = len(xls)
            if len(xls) == 0:
                stackdat['sub'][isub]['profcb'] = np.zeros(Nbins)
                stackdat['sub'][isub]['profps'] = np.zeros(Nbins)
                stackdat['sub'][isub]['profhit'] = np.zeros(Nbins)
                stackdat['sub'][isub]['profcbsub'] = np.zeros(Nsubbins)
                stackdat['sub'][isub]['profpssub'] = np.zeros(Nsubbins)
                stackdat['sub'][isub]['profhitsub'] = np.zeros(Nsubbins)
                stackdat['sub'][isub]['profcb100'] = 0
                stackdat['sub'][isub]['profps100'] = 0
                stackdat['sub'][isub]['profhit100'] = 0
                continue

            xss = np.round(xls * 10 + 4.5).astype(np.int32)
            yss = np.round(yls * 10 + 4.5).astype(np.int32)
            ms = srcdat['sub'][isub]['m' + srctype + '_arr']
            rs = get_mask_radius_th(ifield, ms) # arcsec

            print('stacking %s %d < m < %d, #%d / %d subsample, %d sources, t = %.2f min'\
              %(fieldnamedict[ifield], m_min, m_max,isub, srcdat['Nsub'],\
               len(xls), (time.time()-start_time)/60))

            cbmapstacki, psmapstacki, maskstacki = np.zeros([2*dx+1, 2*dx+1]), \
            np.zeros([2*dx+1, 2*dx+1]), np.zeros([2*dx+1, 2*dx+1])
            for i,(xl,yl,xs,ys,r) in enumerate(zip(xls,yls,xss,yss,rs)):
                if len(xls)>20:
                    if verbose and i%(len(xls)//20)==0:
                        print('stacking %d / %d (%.1f %%), t = %.2f min'\
                              %(i, len(xls), i/len(xls)*100,  (time.time()-start_time)/60))
                cbmapi = cbmap*strmask*mask_inst
                psmapi = psmap*strmask*mask_inst
                maski = strmask*mask_inst
                radmap = make_radius_map(cbmap, xl,yl) # large pix units
                sp1 = np.where((radmap < r/7) & (strnum==1) & (mask_inst==1))
                cbmapi[sp1] = cbmap[sp1]
                psmapi[sp1] = psmap[sp1]
                unmaskpix = np.zeros_like(strmask)
                unmaskpix[sp1] = 1
                maski[sp1] = 1
                if len(sp1[0])>0 and unmask:
                    for ibin in range(Nbins):
                        if cliplim['CBmax'][ibin] == np.inf:
                            continue
                        spi = np.where((unmaskpix==1) & \
                                       (radmap*10>=rbinedges[ibin]) & \
                                       (radmap*10 < rbinedges[ibin+1]) & \
                                       (cbmap > cliplim['CBmax'][ibin]))
                        cbmapi[spi] = 0
                        psmapi[spi] = 0
                        maski[spi] = 0
                        spi = np.where((unmaskpix==1) & \
                                       (radmap*10>=rbinedges[ibin]) & \
                                       (radmap*10 < rbinedges[ibin+1]) & \
                                       (cbmap < cliplim['CBmin'][ibin]))
                        cbmapi[spi] = 0
                        psmapi[spi] = 0
                        maski[spi] = 0
                        spi = np.where((unmaskpix==1) & \
                                       (radmap*10>=rbinedges[ibin]) & \
                                       (radmap*10 < rbinedges[ibin+1]) & \
                                       (psmap > cliplim['PSmax'][ibin]))
                        cbmapi[spi] = 0
                        psmapi[spi] = 0
                        maski[spi] = 0
                        spi = np.where((unmaskpix==1) & \
                                       (radmap*10>=rbinedges[ibin]) & \
                                       (radmap*10 < rbinedges[ibin+1]) & \
                                       (psmap < cliplim['PSmin'][ibin]))
                        cbmapi[spi] = 0
                        psmapi[spi] = 0
                        maski[spi] = 0


                # unmask source
                mcb = cbmapi * maski
                mps = psmapi * maski

                mcb = self._image_finegrid(mcb)
                mps = self._image_finegrid(mps)
                k = self._image_finegrid(maski)

                # zero padding
                mcb = np.pad(mcb, ((dx,dx),(dx,dx)), 'constant')
                mps = np.pad(mps, ((dx,dx),(dx,dx)), 'constant')
                k = np.pad(k, ((dx,dx),(dx,dx)), 'constant')
                xs += dx
                ys += dx

                # cut stamp
                cbmapstamp = mcb[xs - dx: xs + dx + 1, ys - dx: ys + dx + 1]
                psmapstamp = mps[xs - dx: xs + dx + 1, ys - dx: ys + dx + 1]
                maskstamp = k[xs - dx: xs + dx + 1, ys - dx: ys + dx + 1]


                cbmapstacki += cbmapstamp
                psmapstacki += psmapstamp
                maskstacki += maskstamp
            
            ### end source for loop ###
            
            cbmapstack += cbmapstacki
            psmapstack += psmapstacki
            maskstack += maskstacki
                
            profcb_arr, profps_arr, hit_arr \
            = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
            for ibin in range(Nbins):
                spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                               (radmapstamp<rbinedges[ibin+1]))
                profcb_arr[ibin] += np.sum(cbmapstacki[spi])
                profps_arr[ibin] += np.sum(psmapstacki[spi])
                hit_arr[ibin] += np.sum(maskstacki[spi])
            spbin = np.where(hit_arr!=0)[0]
            profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
            profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
            profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]
            stackdat['sub'][isub]['profcb'] = profcb_norm
            stackdat['sub'][isub]['profps'] = profps_norm
            stackdat['sub'][isub]['profhit'] = hit_arr
        
            profcb_arr, profps_arr, hit_arr \
            = np.zeros(Nsubbins), np.zeros(Nsubbins), np.zeros(Nsubbins)
            for ibin in range(Nsubbins):
                spi = np.where((radmapstamp>=rsubbinedges[ibin]) &\
                               (radmapstamp<rsubbinedges[ibin+1]))
                profcb_arr[ibin] += np.sum(cbmapstacki[spi])
                profps_arr[ibin] += np.sum(psmapstacki[spi])
                hit_arr[ibin] += np.sum(maskstacki[spi])
            spbin = np.where(hit_arr!=0)[0]
            profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
            profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
            profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]
            stackdat['sub'][isub]['profcbsub'] = profcb_norm
            stackdat['sub'][isub]['profpssub'] = profps_norm
            stackdat['sub'][isub]['profhitsub'] = hit_arr

            spi = np.where(radmapstamp>=100/0.7)
            if np.sum(maskstacki[spi])!=0:
                stackdat['sub'][isub]['profcb100'] \
                = np.sum(cbmapstacki[spi]) / np.sum(maskstacki[spi])
                stackdat['sub'][isub]['profps100'] \
                = np.sum(psmapstacki[spi]) / np.sum(maskstacki[spi])
                stackdat['sub'][isub]['profhit100'] = np.sum(maskstacki[spi])
            else:
                stackdat['sub'][isub]['profcb100'] = 0
                stackdat['sub'][isub]['profps100'] = 0
                stackdat['sub'][isub]['profhit100'] = 0

        ### end isub for loop ###
        
        spmap = np.where(maskstack!=0)
        cbmapstack_norm = np.zeros_like(cbmapstack)
        psmapstack_norm = np.zeros_like(psmapstack)
        cbmapstack_norm[spmap] = cbmapstack[spmap]/maskstack[spmap]
        psmapstack_norm[spmap] = psmapstack[spmap]/maskstack[spmap]
        if self.savemaps:
            stackdat['cbmapstack'] = cbmapstack_norm
            stackdat['psmapstack'] = psmapstack_norm
            stackdat['maskstack'] = maskstack

        profcb_arr, profps_arr, hit_arr \
        = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
        for ibin in range(Nbins):
            spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                           (radmapstamp<rbinedges[ibin+1]))
            profcb_arr[ibin] += np.sum(cbmapstack[spi])
            profps_arr[ibin] += np.sum(psmapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        spbin = np.where(hit_arr!=0)[0]
        profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
        profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
        profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]
        stackdat['profcb'] = profcb_norm
        stackdat['profps'] = profps_norm
        stackdat['profhit'] = hit_arr

        profcb_arr, profps_arr, hit_arr \
        = np.zeros(Nsubbins), np.zeros(Nsubbins), np.zeros(Nsubbins)
        for ibin in range(Nsubbins):
            spi = np.where((radmapstamp>=rsubbinedges[ibin]) &\
                           (radmapstamp<rsubbinedges[ibin+1]))
            profcb_arr[ibin] += np.sum(cbmapstack[spi])
            profps_arr[ibin] += np.sum(psmapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        spbin = np.where(hit_arr!=0)[0]
        profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
        profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
        profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]
        stackdat['profcbsub'] = profcb_norm
        stackdat['profpssub'] = profps_norm
        stackdat['profhitsub'] = hit_arr

        spi = np.where(radmapstamp>=100/0.7)
        stackdat['profcb100'] = np.sum(cbmapstack[spi]) / np.sum(maskstack[spi])
        stackdat['profps100'] = np.sum(psmapstack[spi]) / np.sum(maskstack[spi])
        stackdat['profhit100'] = np.sum(maskstack[spi])

        return stackdat        
        
    def _stackihl_PS_cliplim(self, Nsrc=np.inf):
        inst = self.inst
        ifield = self.ifield
        m_min = self.m_min
        m_max = self.m_max

        if self.data_maps is None:
            data_maps = {1: image_reduction(1), 2: image_reduction(2)}
            self.data_maps = data_maps
        else:
            data_maps = self.data_maps
        cbmap, psmap, strmask, strnum, mask_inst1, mask_inst2 = \
        load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                            (inst,ifield,'psmap'),
                                            (inst,ifield,'strmask'),
                                           (inst,ifield,'strnum'),
                                           (1,ifield,'mask_inst'),
                                           (2,ifield,'mask_inst')])

        srcdat = ps_src_select(inst, ifield, m_min, m_max, 
            [mask_inst1, mask_inst2], sample_type='all')

        x_arr = np.append(srcdat['xg_arr'],srcdat['xs_arr'])
        y_arr = np.append(srcdat['yg_arr'],srcdat['ys_arr'])
        m_arr = np.append(srcdat['mg_arr'],srcdat['ms_arr'])
        if inst==1:
            mask_inst = mask_inst1
        else:
            mask_inst = mask_inst2

        cbmap = image_poly_filter(cbmap, strmask*mask_inst, degree=self.filt_order)

        r_arr = get_mask_radius_th(ifield, m_arr) # arcsec

        if len(m_arr)>Nsrc:
            sp = np.arange(len(m_arr))
            np.random.shuffle(sp)
            sp = sp[:Nsrc]
        else:
            sp = np.arange(len(m_arr))
        x_arr, y_arr, m_arr, r_arr = x_arr[sp], y_arr[sp], m_arr[sp], r_arr[sp]

        nbins = 25
        dx = 1200
        profile = radial_prof(np.ones([2*dx+1,2*dx+1]), dx, dx)
        rbinedges, rbins = profile['rbinedges'], profile['rbins'] # subpix units

        cbdata, psdata = {}, {}
        for i in range(len(rbins)):
            cbdata[i] = np.array([])
            psdata[i] = np.array([])

        for isrc in range(len(x_arr)):
            radmap = make_radius_map(cbmap, x_arr[isrc], y_arr[isrc]) # large pix units
            sp1 = np.where((radmap < r_arr[isrc]/7) & (strnum==1) & (mask_inst==1))
            if len(sp1[0])==0:
                continue

            # unmasked radii and their CB, PS map values
            ri = radmap[sp1]*10 # sub pix units
            cbi, psi = cbmap[sp1], psmap[sp1]

            for ibin in range(len(rbins)):
                spi = np.where((ri>rbinedges[ibin]) & (ri<rbinedges[ibin+1]))[0]
                if len(spi)==0:
                    continue
                cbdata[ibin] = np.append(cbdata[ibin], cbi[spi])
                psdata[ibin] = np.append(psdata[ibin], psi[spi])

        cliplim = {'rbins': rbins*0.7, 'rbinedges': rbinedges*0.7,
                  'CBmax': np.full((nbins), np.inf),
                  'CBmin': np.full((nbins), -np.inf),
                  'PSmax': np.full((nbins), np.inf),
                  'PSmin': np.full((nbins), -np.inf),
                  }

        d = np.concatenate((cbdata[0],cbdata[1],cbdata[2],cbdata[3]))
        Q1, Q3 = np.percentile(d, 25), np.percentile(d, 75)
        IQR = Q3 - Q1
        cliplim['CBmin'][:4], cliplim['CBmax'][:4]= Q1-3*IQR, Q3+3*IQR

        d = np.concatenate((psdata[0],psdata[1],psdata[2],psdata[3]))
        Q1, Q3 = np.percentile(d, 25), np.percentile(d, 75)
        IQR = Q3 - Q1
        cliplim['PSmin'][:4], cliplim['PSmax'][:4]= Q1-3*IQR, Q3+3*IQR

        for ibin in np.arange(4,nbins,1):
            d = cbdata[ibin]
            if len(d)==0:
                continue
            Q1, Q3 = np.percentile(d, 25), np.percentile(d, 75)
            IQR = Q3 - Q1
            cliplim['CBmin'][ibin], cliplim['CBmax'][ibin]= Q1-3*IQR, Q3+3*IQR
            d = psdata[ibin]
            Q1, Q3 = np.percentile(d, 25), np.percentile(d, 75)
            IQR = Q3 - Q1
            cliplim['PSmin'][ibin], cliplim['PSmax'][ibin]= Q1-3*IQR, Q3+3*IQR

        return cliplim        

    def _image_finegrid(self, image, Nsub=10):
        w, h  = np.shape(image)
        image_new = np.zeros([w*Nsub, h*Nsub])
        for i in range(Nsub):
            for j in range(Nsub):
                image_new[i::Nsub, j::Nsub] = image
        return image_new
    
    def _radial_binning(self,rbins,rbinedges):
        rsubbinedges = np.concatenate((rbinedges[:1],rbinedges[6:20],rbinedges[-1:]))

        # calculate the mean r in and out
        rin = (2./3) * (rsubbinedges[1]**3 - rsubbinedges[0]**3)\
        / (rsubbinedges[1]**2 - rsubbinedges[0]**2)

        rout = (2./3) * (rsubbinedges[-1]**3 - rsubbinedges[-2]**3)\
        / (rsubbinedges[-1]**2 - rsubbinedges[-2]**2)

        rsubbins = np.concatenate(([rin],rbins[6:19],[rout]))

        return rsubbins, rsubbinedges
    
    def _get_jackknife_profile(self):
        self.stackdat['jack'] = {}
        for isub in range(self.stackdat['Nsub']):
            self.stackdat['jack'][isub] = {}
            profcb = self.stackdat['profcb']*self.stackdat['profhit'] - \
            self.stackdat['sub'][isub]['profcb']*self.stackdat['sub'][isub]['profhit']
            profps = self.stackdat['profps']*self.stackdat['profhit'] - \
            self.stackdat['sub'][isub]['profps']*self.stackdat['sub'][isub]['profhit']
            profhit = self.stackdat['profhit'] - self.stackdat['sub'][isub]['profhit']
            self.stackdat['jack'][isub]['profcb'] = profcb/profhit
            self.stackdat['jack'][isub]['profps'] = profps/profhit
            self.stackdat['jack'][isub]['profhit'] = profhit

            profcbsub = self.stackdat['profcbsub']*self.stackdat['profhitsub'] - \
            self.stackdat['sub'][isub]['profcbsub']*self.stackdat['sub'][isub]['profhitsub']
            profpssub = self.stackdat['profpssub']*self.stackdat['profhitsub'] - \
            self.stackdat['sub'][isub]['profpssub']*self.stackdat['sub'][isub]['profhitsub']
            profhitsub = self.stackdat['profhitsub'] - \
            self.stackdat['sub'][isub]['profhitsub']
            self.stackdat['jack'][isub]['profcbsub'] = profcbsub/profhitsub
            self.stackdat['jack'][isub]['profpssub'] = profpssub/profhitsub
            self.stackdat['jack'][isub]['profhitsub'] = profhitsub

            profcb100 = self.stackdat['profcb100']*self.stackdat['profhit100'] - \
            self.stackdat['sub'][isub]['profcb100']*self.stackdat['sub'][isub]['profhit100']
            profps100 = self.stackdat['profps100']*self.stackdat['profhit100'] - \
            self.stackdat['sub'][isub]['profps100']*self.stackdat['sub'][isub]['profhit100']
            profhit100 = self.stackdat['profhit100'] - \
            self.stackdat['sub'][isub]['profhit100']
            self.stackdat['jack'][isub]['profcb100'] = profcb100/profhit100
            self.stackdat['jack'][isub]['profps100'] = profps100/profhit100
            self.stackdat['jack'][isub]['profhit100'] = profhit100
            
        return
    
    def _normalize_cov(self, cov):
        cov_rho = np.zeros_like(cov)
        for i in range(cov_rho.shape[0]):
            for j in range(cov_rho.shape[0]):
                if cov[i,i]==0 or cov[j,j]==0:
                    cov_rho[i,j] = cov[i,j]
                else:
                    cov_rho[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
        return cov_rho

    def _get_covariance(self):
        self.stackdat['cov'] = {}
        Nsub = self.stackdat['Nsub']
        Nbins = len(self.stackdat['rbins'])
        Nsubbins = len(self.stackdat['rsubbins'])
        data_cb, data_ps = np.zeros([Nsub, Nbins]), np.zeros([Nsub, Nbins])
        data_cbsub, data_pssub = np.zeros([Nsub, Nsubbins]), np.zeros([Nsub, Nsubbins])
        data_cb100, data_ps100 = np.zeros(Nsub), np.zeros(Nsub)

        for isub in range(Nsub):
            data_cb[isub,:] = self.stackdat['jack'][isub]['profcb']
            data_ps[isub,:] = self.stackdat['jack'][isub]['profps']
            data_cbsub[isub,:] = self.stackdat['jack'][isub]['profcbsub']
            data_pssub[isub,:] = self.stackdat['jack'][isub]['profpssub']
            data_cb100[isub] = self.stackdat['jack'][isub]['profcb100']
            data_ps100[isub] = self.stackdat['jack'][isub]['profps100']

        covcb = np.zeros([Nbins, Nbins])
        covps = np.zeros([Nbins, Nbins])
        for i in range(Nbins):
            for j in range(Nbins):
                datai, dataj = data_cb[:,i], data_cb[:,j]
                covcb[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
                datai, dataj = data_ps[:,i], data_ps[:,j]
                covps[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
        self.stackdat['cov']['profcb'] = covcb * (Nsub-1)
        self.stackdat['cov']['profps'] = covps * (Nsub-1)
        self.stackdat['cov']['profcb_rho'] = self._normalize_cov(covcb)
        self.stackdat['cov']['profps_rho'] = self._normalize_cov(covps)

        covcb = np.zeros([Nsubbins, Nsubbins])
        covps = np.zeros([Nsubbins, Nsubbins])
        for i in range(Nsubbins):
            for j in range(Nsubbins):
                datai, dataj = data_cbsub[:,i], data_cbsub[:,j]
                covcb[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
                datai, dataj = data_pssub[:,i], data_pssub[:,j]
                covps[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
        self.stackdat['cov']['profcbsub'] = covcb * (Nsub-1)
        self.stackdat['cov']['profpssub'] = covps * (Nsub-1)
        self.stackdat['cov']['profcbsub_rho'] = self._normalize_cov(covcb)
        self.stackdat['cov']['profpssub_rho'] = self._normalize_cov(covps)

        covcb = np.mean(data_cb100**2) - np.mean(data_cb100)**2
        covps = np.mean(data_ps100**2) - np.mean(data_ps100)**2
        self.stackdat['cov']['profcb100'] = covcb * (Nsub-1)
        self.stackdat['cov']['profps100'] = covps * (Nsub-1)

        return
    
    def stack_BG(self, srctype='g', dx=120, verbose=True, Nbg=None, uniform=False):

        inst = self.inst
        ifield = self.ifield
        m_min, m_max = self.m_min, self.m_max
        if Nbg is None:
            Nbg = self.stackdat['Nsub']
        if Nbg < 64:
            uniform = True

        if self.data_maps is None:
            data_maps = {1: image_reduction(1), 2: image_reduction(2)}
            self.data_maps = data_maps
        else:
            data_maps = self.data_maps
        cbmap, psmap, strmask, mask_inst = \
        load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                            (inst,ifield,'psmap'),
                                           (inst,ifield,'strmask'), 
                                           (inst,ifield,'mask_inst')])
        cbmap = image_poly_filter(cbmap, strmask*mask_inst, degree=self.filt_order)

        Nsrc = self.stackdat['Nsrc']
        self.stackdat['BG'] = {}
        self.stackdat['BG']['Nbg'] = Nbg
        
        # start stacking
        Nbins = len(self.stackdat['rbins'])
        Nsubbins = len(self.stackdat['rsubbins'])
        radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx) # pix unit
        rbins = self.stackdat['rbins']/7 # pix unit
        rsubbins = self.stackdat['rsubbins']/7 # pix unit        
        rbinedges = self.stackdat['rbinedges']/7 # pix unit
        rsubbinedges = self.stackdat['rsubbinedges']/7 # pix unit

        cbmapi = cbmap*strmask*mask_inst
        psmapi = psmap*strmask*mask_inst
        maski = strmask*mask_inst
        
        Nsides = int(np.sqrt(Nbg))
        axlims = np.linspace(-0.5, 1023.5, Nsides+1)
        ymins, xmins = np.meshgrid(axlims[:-1], axlims[:-1])
        ymaxs, xmaxs = np.meshgrid(axlims[1:], axlims[1:])
    
        cbmapstack, psmapstack, maskstack = 0., 0., 0
        start_time = time.time()
        for isub in range(Nbg):
            self.stackdat['BG'][isub] = {}
            Nsrc = self.stackdat['sub'][isub]['Nsrc']
            print('stacking %s %d < m < %d, #%d BG, %d sources, t = %.2f min'\
              %(fieldnamedict[ifield], m_min, m_max,\
                isub, Nsrc, (time.time()-start_time)/60))
            
            if Nsrc == 0:
                self.stackdat['BG'][isub]['profcb'] = np.zeros(Nbins)
                self.stackdat['BG'][isub]['profps'] = np.zeros(Nbins)
                self.stackdat['BG'][isub]['profhit'] = np.zeros(Nbins)
                self.stackdat['BG'][isub]['profcbsub'] = np.zeros(Nsubbins)
                self.stackdat['BG'][isub]['profpssub'] = np.zeros(Nsubbins)
                self.stackdat['BG'][isub]['profhitsub'] = np.zeros(Nsubbins)
                self.stackdat['BG'][isub]['profcb100'] = 0
                self.stackdat['BG'][isub]['profps100'] = 0
                self.stackdat['BG'][isub]['profhit100'] = 0
                continue

            if uniform:
                xs = np.random.randint(-0.5,1023.5,Nsrc)
                ys = np.random.randint(-0.5,1023.5,Nsrc)
            else:
                ymin, xmin = ymins.flatten()[isub], xmins.flatten()[isub]
                ymax, xmax = ymaxs.flatten()[isub], xmaxs.flatten()[isub]
                xs = np.random.randint(xmin,xmax,Nsrc)
                ys = np.random.randint(ymin,ymax,Nsrc)

            cbmapstacki, psmapstacki, maskstacki = 0., 0., 0
            for i,(xi,yi) in enumerate(zip(xs,ys)):
                
                radmap = make_radius_map(cbmap, xi, yi) # large pix units

                # zero padding
                mcb = np.pad(cbmapi, ((dx,dx),(dx,dx)), 'constant')
                mps = np.pad(psmapi, ((dx,dx),(dx,dx)), 'constant')
                k = np.pad(maski, ((dx,dx),(dx,dx)), 'constant')
                xi += dx
                yi += dx

                # cut stamp
                cbmapstamp = mcb[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]
                psmapstamp = mps[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]
                maskstamp = k[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]

                cbmapstacki += cbmapstamp
                psmapstacki += psmapstamp
                maskstacki += maskstamp

            ### end source for loop ###
            cbmapstack += cbmapstacki
            psmapstack += psmapstacki
            maskstack += maskstacki

            profcb_arr, profps_arr, hit_arr \
            = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
            for ibin in range(Nbins):
                spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                               (radmapstamp<rbinedges[ibin+1]))
                profcb_arr[ibin] += np.sum(cbmapstacki[spi])
                profps_arr[ibin] += np.sum(psmapstacki[spi])
                hit_arr[ibin] += np.sum(maskstacki[spi])
            spbin = np.where(hit_arr!=0)[0]
            profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
            profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
            profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]
            profcb_norm = np.interp(rbins, rbins[spbin], profcb_norm[spbin])
            profps_norm = np.interp(rbins, rbins[spbin], profps_norm[spbin])
            self.stackdat['BG'][isub]['profcb'] = profcb_norm
            self.stackdat['BG'][isub]['profps'] = profps_norm
            self.stackdat['BG'][isub]['profhit'] = hit_arr

            profcb_arr, profps_arr, hit_arr \
            = np.zeros(Nsubbins), np.zeros(Nsubbins), np.zeros(Nsubbins)
            for ibin in range(Nsubbins):
                spi = np.where((radmapstamp>=rsubbinedges[ibin]) &\
                               (radmapstamp<rsubbinedges[ibin+1]))
                profcb_arr[ibin] += np.sum(cbmapstacki[spi])
                profps_arr[ibin] += np.sum(psmapstacki[spi])
                hit_arr[ibin] += np.sum(maskstacki[spi])
            spbin = np.where(hit_arr!=0)[0]
            profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
            profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
            profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]
            profcb_norm = np.interp(rsubbins, rsubbins[spbin], profcb_norm[spbin])
            profps_norm = np.interp(rsubbins, rsubbins[spbin], profps_norm[spbin])
            self.stackdat['BG'][isub]['profcbsub'] = profcb_norm
            self.stackdat['BG'][isub]['profpssub'] = profps_norm
            self.stackdat['BG'][isub]['profhitsub'] = hit_arr

            spi = np.where(radmapstamp>=100/7)
            self.stackdat['BG'][isub]['profcb100'] \
            = np.sum(cbmapstacki[spi]) / np.sum(maskstacki[spi])
            self.stackdat['BG'][isub]['profps100'] \
            = np.sum(psmapstacki[spi]) / np.sum(maskstacki[spi])
            self.stackdat['BG'][isub]['profhit100'] = np.sum(maskstacki[spi])

        ### end isub for loop ###

        spmap = np.where(maskstack!=0)
        cbmapstack_norm = np.zeros_like(cbmapstack)
        psmapstack_norm = np.zeros_like(psmapstack)
        cbmapstack_norm[spmap] = cbmapstack[spmap]/maskstack[spmap]
        psmapstack_norm[spmap] = psmapstack[spmap]/maskstack[spmap]
        if self.savemaps:
            self.stackdat['cbmapstackBG'] = cbmapstack_norm
            self.stackdat['psmapstackBG'] = psmapstack_norm
            self.stackdat['maskstackBG'] = maskstack
        
        profcb_arr, profps_arr, hit_arr \
        = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
        for ibin in range(Nbins):
            spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                           (radmapstamp<rbinedges[ibin+1]))
            profcb_arr[ibin] += np.sum(cbmapstack[spi])
            profps_arr[ibin] += np.sum(psmapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        spbin = np.where(hit_arr!=0)[0]
        profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
        profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
        profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]
        self.stackdat['BG']['profcb'] = profcb_norm
        self.stackdat['BG']['profps'] = profps_norm
        self.stackdat['BG']['profhit'] = hit_arr

        profcb_arr, profps_arr, hit_arr \
        = np.zeros(Nsubbins), np.zeros(Nsubbins), np.zeros(Nsubbins)
        for ibin in range(Nsubbins):
            spi = np.where((radmapstamp>=rsubbinedges[ibin]) &\
                           (radmapstamp<rsubbinedges[ibin+1]))
            profcb_arr[ibin] += np.sum(cbmapstack[spi])
            profps_arr[ibin] += np.sum(psmapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        spbin = np.where(hit_arr!=0)[0]
        profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
        profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
        profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]
        self.stackdat['BG']['profcbsub'] = profcb_norm
        self.stackdat['BG']['profpssub'] = profps_norm
        self.stackdat['BG']['profhitsub'] = hit_arr

        spi = np.where(radmapstamp>=100/0.7)
        self.stackdat['BG']['profcb100'] = np.sum(cbmapstack[spi]) / np.sum(maskstack[spi])
        self.stackdat['BG']['profps100'] = np.sum(psmapstack[spi]) / np.sum(maskstack[spi])
        self.stackdat['BG']['profhit100'] = np.sum(maskstack[spi])

        return

    def _get_BG_jackknife_profile(self):
        self.stackdat['BGjack'] = {}
        for isub in range(self.stackdat['BG']['Nbg']):
            self.stackdat['BGjack'][isub] = {}
            profcb = self.stackdat['BG']['profcb']*self.stackdat['BG']['profhit'] - \
            self.stackdat['BG'][isub]['profcb']*self.stackdat['BG'][isub]['profhit']
            profps = self.stackdat['BG']['profps']*self.stackdat['BG']['profhit'] - \
            self.stackdat['BG'][isub]['profps']*self.stackdat['BG'][isub]['profhit']
            profhit = self.stackdat['BG']['profhit'] - self.stackdat['BG'][isub]['profhit']
            
            sp = np.where(profhit!=0)[0]
            p = np.zeros_like(profcb)
            p[sp] = profcb[sp]/profhit[sp]
            self.stackdat['BGjack'][isub]['profcb'] = p
            p = np.zeros_like(profps)
            p[sp] = profps[sp]/profhit[sp]
            self.stackdat['BGjack'][isub]['profps'] = p
            self.stackdat['BGjack'][isub]['profhit'] = profhit

            profcbsub = self.stackdat['BG']['profcbsub']*self.stackdat['BG']['profhitsub'] - \
            self.stackdat['BG'][isub]['profcbsub']*self.stackdat['BG'][isub]['profhitsub']
            profpssub = self.stackdat['BG']['profpssub']*self.stackdat['BG']['profhitsub'] - \
            self.stackdat['BG'][isub]['profpssub']*self.stackdat['BG'][isub]['profhitsub']
            profhitsub = self.stackdat['BG']['profhitsub'] - \
            self.stackdat['BG'][isub]['profhitsub']
            
            sp = np.where(profhitsub!=0)[0]
            p = np.zeros_like(profcbsub)
            p[sp] = profcbsub[sp]/profhitsub[sp]
            self.stackdat['BGjack'][isub]['profcbsub'] = p
            p = np.zeros_like(profpssub)
            p[sp] = profpssub[sp]/profhitsub[sp]
            self.stackdat['BGjack'][isub]['profpssub'] = p
            self.stackdat['BGjack'][isub]['profhitsub'] = profhitsub

            profcb100 = self.stackdat['BG']['profcb100']*self.stackdat['BG']['profhit100'] - \
            self.stackdat['BG'][isub]['profcb100']*self.stackdat['BG'][isub]['profhit100']
            profps100 = self.stackdat['BG']['profps100']*self.stackdat['BG']['profhit100'] - \
            self.stackdat['BG'][isub]['profps100']*self.stackdat['BG'][isub]['profhit100']
            profhit100 = self.stackdat['BG']['profhit100'] - \
            self.stackdat['BG'][isub]['profhit100']

            if profhit100 == 0:
                self.stackdat['BGjack'][isub]['profcb100'] = 0
                self.stackdat['BGjack'][isub]['profps100'] = 0
            else:
                self.stackdat['BGjack'][isub]['profcb100'] = profcb100/profhit100
                self.stackdat['BGjack'][isub]['profps100'] = profps100/profhit100
            self.stackdat['BGjack'][isub]['profhit100'] = profhit100
            
        return

    def _get_BG_covariance(self):
        self.stackdat['BGcov'] = {}
        Nsub = self.stackdat['BG']['Nbg']
        Nbins = len(self.stackdat['rbins'])
        Nsubbins = len(self.stackdat['rsubbins'])
        data_cb, data_ps = np.zeros([Nsub, Nbins]), np.zeros([Nsub, Nbins])
        data_cbsub, data_pssub = np.zeros([Nsub, Nsubbins]), np.zeros([Nsub, Nsubbins])
        data_cb100, data_ps100 = np.zeros(Nsub), np.zeros(Nsub)

        for isub in range(Nsub):
            data_cb[isub,:] = self.stackdat['BGjack'][isub]['profcb']
            data_ps[isub,:] = self.stackdat['BGjack'][isub]['profps']
            data_cbsub[isub,:] = self.stackdat['BGjack'][isub]['profcbsub']
            data_pssub[isub,:] = self.stackdat['BGjack'][isub]['profpssub']
            data_cb100[isub] = self.stackdat['BGjack'][isub]['profcb100']
            data_ps100[isub] = self.stackdat['BGjack'][isub]['profps100']

        covcb = np.zeros([Nbins, Nbins])
        covps = np.zeros([Nbins, Nbins])
        for i in range(Nbins):
            for j in range(Nbins):
                datai, dataj = data_cb[:,i], data_cb[:,j]
                covcb[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
                datai, dataj = data_ps[:,i], data_ps[:,j]
                covps[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
        self.stackdat['BGcov']['profcb'] = covcb * (Nsub-1)
        self.stackdat['BGcov']['profps'] = covps * (Nsub-1)
        self.stackdat['BGcov']['profcb_rho'] = self._normalize_cov(covcb)
        self.stackdat['BGcov']['profps_rho'] = self._normalize_cov(covps)

        covcb = np.zeros([Nsubbins, Nsubbins])
        covps = np.zeros([Nsubbins, Nsubbins])
        for i in range(Nsubbins):
            for j in range(Nsubbins):
                datai, dataj = data_cbsub[:,i], data_cbsub[:,j]
                covcb[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
                datai, dataj = data_pssub[:,i], data_pssub[:,j]
                covps[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
        self.stackdat['BGcov']['profcbsub'] = covcb * (Nsub-1)
        self.stackdat['BGcov']['profpssub'] = covps * (Nsub-1)
        self.stackdat['BGcov']['profcbsub_rho'] = self._normalize_cov(covcb)
        self.stackdat['BGcov']['profpssub_rho'] = self._normalize_cov(covps)

        covcb = np.mean(data_cb100**2) - np.mean(data_cb100)**2
        covps = np.mean(data_ps100**2) - np.mean(data_ps100)**2
        self.stackdat['BGcov']['profcb100'] = covcb * (Nsub-1)
        self.stackdat['BGcov']['profps100'] = covps * (Nsub-1)

    def _get_BGsub(self):
        self.stackdat['BGsub'] = {}

        if self.BGsub:
            self.stackdat['BGsub']['profcb'] = self.stackdat['profcb'] \
            - self.stackdat['BG']['profcb']
            self.stackdat['BGsub']['profps'] = self.stackdat['profps'] \
            - self.stackdat['BG']['profps']
            self.stackdat['BGsub']['profcbsub'] = self.stackdat['profcbsub'] \
            - self.stackdat['BG']['profcbsub']
            self.stackdat['BGsub']['profpssub'] = self.stackdat['profpssub'] \
            - self.stackdat['BG']['profpssub']
            self.stackdat['BGsub']['profcb100'] = self.stackdat['profcb100'] \
            - self.stackdat['BG']['profcb100']
            self.stackdat['BGsub']['profps100'] = self.stackdat['profps100'] \
            - self.stackdat['BG']['profps100']
        else:
            self.stackdat['BGsub']['profcb'] = self.stackdat['profcb'].copy()
            self.stackdat['BGsub']['profps'] = self.stackdat['profps'].copy()
            self.stackdat['BGsub']['profcbsub'] = self.stackdat['profcbsub'].copy()
            self.stackdat['BGsub']['profpssub'] = self.stackdat['profpssub'].copy()
            self.stackdat['BGsub']['profcb100'] = self.stackdat['profcb100'].copy()
            self.stackdat['BGsub']['profps100'] = self.stackdat['profps100'].copy()

    def _get_PSF_from_data(self):

        scalecb = self.stackdat['BGsub']['profcb'][0]
        scaleps = self.stackdat['BGsub']['profps'][0]
        self.stackdat['PSF'] = {}

        fname = mypaths['alldat'] + 'TM'+ str(self.inst) + \
        '/psfdata_synth_%s.pkl'%(self.field)
        with open(fname, "rb") as f:
            profdat = pickle.load(f)

        im = self.m_min-16
        if im in profdat:
            self.stackdat['PSF']['Nsrc'] = profdat[self.m_min-16]['Nsrc']
            psfdat = profdat[im]['comb']
            self.stackdat['PSF']['profcb'] = psfdat['profcb']*scalecb
            self.stackdat['PSF']['profps'] = psfdat['profcb']*scaleps
            self.stackdat['PSF']['profcbsub'] = psfdat['profcbsub']*scalecb
            self.stackdat['PSF']['profpssub'] = psfdat['profcbsub']*scaleps
            self.stackdat['PSF']['profcb100'] = psfdat['profcbsub'][-1]*scalecb
            self.stackdat['PSF']['profps100'] = psfdat['profcbsub'][-1]*scaleps
        else:
            # this is for running PSF stack only (run_psf_synth)
            self.stackdat['PSF']['profcb'] = self.stackdat['profcb']
            self.stackdat['PSF']['profps'] = self.stackdat['profcb']
            self.stackdat['PSF']['profcbsub'] = self.stackdat['profcbsub']
            self.stackdat['PSF']['profpssub'] = self.stackdat['profcbsub']
            self.stackdat['PSF']['profcb100'] = self.stackdat['profcbsub'][-1]
            self.stackdat['PSF']['profps100'] = self.stackdat['profcbsub'][-1]


        # import json
        # loaddir = mypaths['alldat']+'TM' + str(self.inst) + '/'
        # with open(loaddir + self.field + '_datafit.json') as json_file:
        #     data_all = json.load(json_file)
        # im = self.stackdat['m_min'] - 16
        # data = data_all[im]
        # self.stackdat['PSF']['profcb'] = np.array(data['profpsfcbfull'])*scalecb
        # self.stackdat['PSF']['profps'] = np.array(data['profpsfpsfull'])*scaleps
        # self.stackdat['PSF']['profcbsub'] = np.array(data['profpsfcb'])*scalecb
        # self.stackdat['PSF']['profpssub'] = np.array(data['profpsfps'])*scaleps
        # self.stackdat['PSF']['profcb100'] = np.array(data['profpsfcb100'])*scalecb
        # self.stackdat['PSF']['profps100'] = np.array(data['profpsfps100'])*scaleps


    def _get_PSF_covariance_from_data(self):
        scalecb = self.stackdat['BGsub']['profcb'][0]
        scaleps = self.stackdat['BGsub']['profps'][0]
        self.stackdat['PSFcov'] = {}

        fname = mypaths['alldat'] + 'TM'+ str(self.inst) + \
        '/psfdata_synth_%s.pkl'%(self.field)
        with open(fname, "rb") as f:
            profdat = pickle.load(f)

        im = self.m_min-16
        if im in profdat:
            psfdat = profdat[self.m_min-16]['comb']
            self.stackdat['PSFcov']['profcb'] = psfdat['cov']*scalecb**2
            self.stackdat['PSFcov']['profps'] = psfdat['cov']*scalecb**2
            self.stackdat['PSFcov']['profcbsub'] = psfdat['covsub']*scalecb**2
            self.stackdat['PSFcov']['profpssub'] = psfdat['covsub']*scalecb**2
            self.stackdat['PSFcov']['profcb100'] = psfdat['covsub'][-1,-1]*scalecb**2
            self.stackdat['PSFcov']['profps100'] = psfdat['covsub'][-1,-1]*scalecb**2
            self.stackdat['PSFcov']['profcb_rho'] \
            = self._normalize_cov(self.stackdat['PSFcov']['profcb'])
            self.stackdat['PSFcov']['profps_rho'] \
            = self._normalize_cov(self.stackdat['PSFcov']['profps'])
            self.stackdat['PSFcov']['profcbsub_rho'] \
            = self._normalize_cov(self.stackdat['PSFcov']['profcbsub'])
            self.stackdat['PSFcov']['profpssub_rho'] \
            = self._normalize_cov(self.stackdat['PSFcov']['profpssub'])
        else:
            # this is for running PSF stack only (run_psf_synth)
            self.stackdat['PSFcov']['profcb'] = self.stackdat['cov']['profcb']
            self.stackdat['PSFcov']['profps'] = self.stackdat['cov']['profcb']
            self.stackdat['PSFcov']['profcbsub'] = self.stackdat['cov']['profcbsub']
            self.stackdat['PSFcov']['profpssub'] = self.stackdat['cov']['profcbsub']
            self.stackdat['PSFcov']['profcb100'] = self.stackdat['cov']['profcbsub'][-1,-1]
            self.stackdat['PSFcov']['profps100'] = self.stackdat['cov']['profcbsub'][-1,-1]
            self.stackdat['PSFcov']['profcb_rho'] \
            = self._normalize_cov(self.stackdat['PSFcov']['profcb'])
            self.stackdat['PSFcov']['profps_rho'] \
            = self._normalize_cov(self.stackdat['PSFcov']['profps'])
            self.stackdat['PSFcov']['profcbsub_rho'] \
            = self._normalize_cov(self.stackdat['PSFcov']['profcbsub'])
            self.stackdat['PSFcov']['profpssub_rho'] \
            = self._normalize_cov(self.stackdat['PSFcov']['profpssub'])

        # import json
        # loaddir = mypaths['alldat']+'TM' + str(self.inst) + '/'
        # with open(loaddir + self.field + '_datafit.json') as json_file:
        #     data_all = json.load(json_file)
        # im = self.stackdat['m_min'] - 16
        # data = data_all[im]
        # self.stackdat['PSFcov']['profcb'] = np.array(data['covpsfcbfull'])
        # self.stackdat['PSFcov']['profps'] = np.array(data['covpsfpsfull'])
        # self.stackdat['PSFcov']['profcbsub'] = np.array(data['covpsfcb'])
        # self.stackdat['PSFcov']['profpssub'] = np.array(data['covpsfps'])
        # self.stackdat['PSFcov']['profcb100'] = np.array(data['covpsfcb100'])
        # self.stackdat['PSFcov']['profps100'] = np.array(data['covpsfps100'])
        # self.stackdat['PSFcov']['profcb_rho'] \
        # = self._normalize_cov(self.stackdat['PSFcov']['profcb'])
        # self.stackdat['PSFcov']['profps_rho'] \
        # = self._normalize_cov(self.stackdat['PSFcov']['profps'])
        # self.stackdat['PSFcov']['profcbsub_rho'] \
        # = self._normalize_cov(self.stackdat['PSFcov']['profcbsub'])
        # self.stackdat['PSFcov']['profpssub_rho'] \
        # = self._normalize_cov(self.stackdat['PSFcov']['profpssub'])


    def _get_ex_covariance(self):
        if self.BGsub:
            self.stackdat['excov'] = {}
            self.stackdat['excov']['profcb'] = self.stackdat['cov']['profcb'] +\
                self.stackdat['BGcov']['profcb'] + self.stackdat['PSFcov']['profcb']
            self.stackdat['excov']['profps'] = self.stackdat['cov']['profps'] +\
                self.stackdat['BGcov']['profps'] + self.stackdat['PSFcov']['profps']
            self.stackdat['excov']['profcbsub'] = self.stackdat['cov']['profcbsub'] +\
                self.stackdat['BGcov']['profcbsub'] + self.stackdat['PSFcov']['profcbsub']
            self.stackdat['excov']['profpssub'] = self.stackdat['cov']['profpssub'] +\
                self.stackdat['BGcov']['profpssub'] + self.stackdat['PSFcov']['profpssub']
            self.stackdat['excov']['profcb100'] = self.stackdat['cov']['profcb100'] +\
                self.stackdat['BGcov']['profcb100'] + self.stackdat['PSFcov']['profcb100']
            self.stackdat['excov']['profps100'] = self.stackdat['cov']['profps100'] +\
                self.stackdat['BGcov']['profps100'] + self.stackdat['PSFcov']['profps100']
        else:
            self.stackdat['excov'] = {}
            self.stackdat['excov']['profcb'] = self.stackdat['cov']['profcb'] +\
                self.stackdat['PSFcov']['profcb']
            self.stackdat['excov']['profps'] = self.stackdat['cov']['profps'] +\
                self.stackdat['PSFcov']['profps']
            self.stackdat['excov']['profcbsub'] = self.stackdat['cov']['profcbsub'] +\
                self.stackdat['PSFcov']['profcbsub']
            self.stackdat['excov']['profpssub'] = self.stackdat['cov']['profpssub'] +\
                self.stackdat['PSFcov']['profpssub']
            self.stackdat['excov']['profcb100'] = self.stackdat['cov']['profcb100'] +\
                self.stackdat['PSFcov']['profcb100']
            self.stackdat['excov']['profps100'] = self.stackdat['cov']['profps100'] +\
                self.stackdat['PSFcov']['profps100']

        self.stackdat['excov']['profcb_rho'] \
        = self._normalize_cov(self.stackdat['excov']['profcb'])
        self.stackdat['excov']['profps_rho'] \
        = self._normalize_cov(self.stackdat['excov']['profps'])
        self.stackdat['excov']['profcbsub_rho'] \
        = self._normalize_cov(self.stackdat['excov']['profcbsub'])
        self.stackdat['excov']['profpssub_rho'] \
        = self._normalize_cov(self.stackdat['excov']['profpssub'])

    def _get_excess(self):
        self.stackdat['ex'] = {}
        self.stackdat['ex']['profcb'] = self.stackdat['BGsub']['profcb'] \
                                        - self.stackdat['PSF']['profcb']
        self.stackdat['ex']['profps'] = self.stackdat['BGsub']['profps'] \
                                        - self.stackdat['PSF']['profps']
        self.stackdat['ex']['profcbsub'] = self.stackdat['BGsub']['profcbsub'] \
                                        - self.stackdat['PSF']['profcbsub']
        self.stackdat['ex']['profpssub'] = self.stackdat['BGsub']['profpssub'] \
                                        - self.stackdat['PSF']['profpssub']
        self.stackdat['ex']['profcb100'] = self.stackdat['BGsub']['profcb100'] \
                                        - self.stackdat['PSF']['profcb100']
        self.stackdat['ex']['profps100'] = self.stackdat['BGsub']['profps100'] \
                                        - self.stackdat['PSF']['profps100']


    def _get_subsubbins(self, Nrebin=7):
        # further combine Nrebin of rsubbins into one bin
        rsubbins = self.stackdat['rsubbins'].copy()
        rsubbinedges = self.stackdat['rsubbinedges'].copy()
        self.stackdat['rsubbins0'] = rsubbins
        self.stackdat['rsubbinedges0'] = rsubbinedges      
        Nsub = len(rsubbins)

        rsubbinedges = np.concatenate((rsubbinedges[:1],rsubbinedges[Nrebin:]))
        rin = (2./3) * (rsubbinedges[1]**3 - rsubbinedges[0]**3)\
        / (rsubbinedges[1]**2 - rsubbinedges[0]**2)

        rsubbins = np.concatenate(([rin],rsubbins[Nrebin:]))

        self.stackdat['rsubbins'] = rsubbins
        self.stackdat['rsubbinedges'] = rsubbinedges

        Nsubsub = len(rsubbins)

        # mean profile
        profcb,profps, profhit = np.zeros(Nsubsub), np.zeros(Nsubsub), np.zeros(Nsubsub)
        profcb[1:] = self.stackdat['profcbsub'][Nrebin:]
        profps[1:] = self.stackdat['profpssub'][Nrebin:]
        profhit[1:] = self.stackdat['profhitsub'][Nrebin:]
        cbin = self.stackdat['profcbsub'][:Nrebin]
        psin = self.stackdat['profpssub'][:Nrebin]
        hitin = self.stackdat['profhitsub'][:Nrebin]

        if np.sum(hitin)!=0:
            profcb[0] = np.sum(cbin*hitin) / np.sum(hitin)
            profps[0] = np.sum(psin*hitin) / np.sum(hitin)
            profhit[0] = np.sum(hitin)

        self.stackdat['profcbsub'] = profcb
        self.stackdat['profpssub'] = profps
        self.stackdat['profhitsub'] = profhit

        # substack profile
        for isub in range(self.stackdat['Nsub']):
            profcb,profps, profhit = np.zeros(Nsubsub), np.zeros(Nsubsub), np.zeros(Nsubsub)
            profcb[1:] = self.stackdat['sub'][isub]['profcbsub'][Nrebin:]
            profps[1:] = self.stackdat['sub'][isub]['profpssub'][Nrebin:]
            profhit[1:] = self.stackdat['sub'][isub]['profhitsub'][Nrebin:]
            cbin = self.stackdat['sub'][isub]['profcbsub'][:Nrebin]
            psin = self.stackdat['sub'][isub]['profpssub'][:Nrebin]
            hitin = self.stackdat['sub'][isub]['profhitsub'][:Nrebin]

            if np.sum(hitin)!=0:
                profcb[0] = np.sum(cbin*hitin) / np.sum(hitin)
                profps[0] = np.sum(psin*hitin) / np.sum(hitin)
                profhit[0] = np.sum(hitin)

            self.stackdat['sub'][isub]['profcbsub'] = profcb
            self.stackdat['sub'][isub]['profpssub'] = profps
            self.stackdat['sub'][isub]['profhitsub'] = profhit

        # mean BG profile
        profcb,profps, profhit = np.zeros(Nsubsub), np.zeros(Nsubsub), np.zeros(Nsubsub)
        profcb[1:] = self.stackdat['BG']['profcbsub'][Nrebin:]
        profps[1:] = self.stackdat['BG']['profpssub'][Nrebin:]
        profhit[1:] = self.stackdat['BG']['profhitsub'][Nrebin:]
        cbin = self.stackdat['BG']['profcbsub'][:Nrebin]
        psin = self.stackdat['BG']['profpssub'][:Nrebin]
        hitin = self.stackdat['BG']['profhitsub'][:Nrebin]

        if np.sum(hitin)!=0:
            profcb[0] = np.sum(cbin*hitin) / np.sum(hitin)
            profps[0] = np.sum(psin*hitin) / np.sum(hitin)
            profhit[0] = np.sum(hitin)

        self.stackdat['BG']['profcbsub'] = profcb
        self.stackdat['BG']['profpssub'] = profps
        self.stackdat['BG']['profhitsub'] = profhit

        # substack profile
        for isub in range(self.stackdat['Nsub']):
            profcb,profps, profhit = np.zeros(Nsubsub), np.zeros(Nsubsub), np.zeros(Nsubsub)
            profcb[1:] = self.stackdat['BG'][isub]['profcbsub'][Nrebin:]
            profps[1:] = self.stackdat['BG'][isub]['profpssub'][Nrebin:]
            profhit[1:] = self.stackdat['BG'][isub]['profhitsub'][Nrebin:]
            cbin = self.stackdat['BG'][isub]['profcbsub'][:Nrebin]
            psin = self.stackdat['BG'][isub]['profpssub'][:Nrebin]
            hitin = self.stackdat['BG'][isub]['profhitsub'][:Nrebin]

            if np.sum(hitin)!=0:
                profcb[0] = np.sum(cbin*hitin) / np.sum(hitin)
                profps[0] = np.sum(psin*hitin) / np.sum(hitin)
                profhit[0] = np.sum(hitin)

            self.stackdat['BG'][isub]['profcbsub'] = profcb
            self.stackdat['BG'][isub]['profpssub'] = profps
            self.stackdat['BG'][isub]['profhitsub'] = profhit

        return

    def _get_PSF_subsubbins(self, Nrebin=7):
        # further combine Nrebin of rsubbins into one bin
        
        Nsub = len(self.stackdat['PSF']['profcbsub'])
        Nsubsub = Nsub - Nrebin + 1
        profcb,profps= np.zeros(Nsubsub), np.zeros(Nsubsub)
        profcb[1:] = self.stackdat['PSF']['profcbsub'].copy()[Nrebin:]
        profps[1:] = self.stackdat['PSF']['profpssub'].copy()[Nrebin:]

        cbin = self.stackdat['PSF']['profcbsub'].copy()[:Nrebin]
        psin = self.stackdat['PSF']['profpssub'].copy()[:Nrebin]  
        rsubbinedges = self.stackdat['rsubbinedges0'][:Nrebin+1]
        hitin = rsubbinedges[1:]**2 - rsubbinedges[:-1]**2
        profcb[0] = np.sum(cbin*hitin) / np.sum(hitin)
        profps[0] = np.sum(psin*hitin) / np.sum(hitin)

        self.stackdat['PSF']['profcbsub'] = profcb
        self.stackdat['PSF']['profpssub'] = profps
        
        covcb, covps = np.zeros([Nsubsub, Nsubsub]), np.zeros([Nsubsub, Nsubsub])
        covcb[1:,1:] = self.stackdat['PSFcov']['profcbsub'].copy()[Nrebin:, Nrebin:]
        covps[1:,1:] = self.stackdat['PSFcov']['profpssub'].copy()[Nrebin:, Nrebin:]
        covcb[0,1:] = np.sum(self.stackdat['PSFcov']['profcbsub'].copy()[:Nrebin, Nrebin:], axis=0)
        covcb[1:,0] = covcb[0,1:].copy()
        covps[0,1:] = np.sum(self.stackdat['PSFcov']['profpssub'][:Nrebin, Nrebin:], axis=0)
        covps[1:,0] = covps[0,1:].copy()
        covcb[0,0] = np.sum(self.stackdat['PSFcov']['profcbsub'][:Nrebin, :Nrebin])
        covps[0,0] = np.sum(self.stackdat['PSFcov']['profpssub'][:Nrebin, :Nrebin])
        self.stackdat['PSFcov']['profcbsub'] = covcb
        self.stackdat['PSFcov']['profpssub'] = covps
        self.stackdat['PSFcov']['profcbsub_rho'] \
        = self._normalize_cov(self.stackdat['PSFcov']['profcbsub'])
        self.stackdat['PSFcov']['profpssub_rho'] \
        = self._normalize_cov(self.stackdat['PSFcov']['profpssub'])

        return
def run_stacking(inst, ifield, **kwargs):
    for m_min, m_max in zip(magbindict['m_min'], magbindict['m_max']):
        stacking(inst, ifield, m_min, m_max, **kwargs)
    return

from mask import *
import pandas as pd
import time

def ps_src_select(inst, ifield, m_min, m_max, mask_insts,
                  Nsub=50, sample_type='jack_random'):
    catdir = mypaths['PScatdat']
    df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')

    x1_arr, y1_arr = np.array(df['y1']), np.array(df['x1'])
    x2_arr, y2_arr = np.array(df['y2']), np.array(df['x2'])

    cls_arr = np.array(df['sdssClass'])
    cls_arr[cls_arr==3] = 1
    cls_arr[cls_arr==6] = -1

    photz_arr = np.array(df['Photz'])

    m_arr = np.array(df['I_comb'])
    if inst==1:
        m0_arr = np.array(df['I_comb'])
    else:
        m0_arr = np.array(df['H_comb'])

    sp = np.where((x1_arr>-0.5) & (x1_arr<1023.5) & (x2_arr>-0.5) & (x2_arr<1023.5) & \
                 (y1_arr>-0.5) & (y1_arr<1023.5) & (y2_arr>-0.5) & (y2_arr<1023.5))[0]

    x1_arr, y1_arr, x2_arr, y2_arr = x1_arr[sp], y1_arr[sp], x2_arr[sp], y2_arr[sp]
    m_arr, m0_arr, cls_arr, photz_arr = m_arr[sp], m0_arr[sp], cls_arr[sp], photz_arr[sp]

    # count the center pix map
    centnum_map1, _, _ = np.histogram2d(x1_arr, y1_arr, np.arange(-0.5,1024.5,1))
    centnum_map2, _, _ = np.histogram2d(x2_arr, y2_arr, np.arange(-0.5,1024.5,1))

    spg = np.where((m_arr<=m_max) & (m_arr>m_min) & (cls_arr==1) & (photz_arr>=0))[0]
    sps = np.where((m_arr<=m_max) & (m_arr>m_min) & (cls_arr==-1))[0]
    sp = np.append(sps,spg)
    x1_arr, y1_arr, x2_arr, y2_arr = x1_arr[sp], y1_arr[sp], x2_arr[sp], y2_arr[sp]
    m_arr, m0_arr, z_arr = m_arr[sp], m0_arr[sp], photz_arr[sp]
    cls_arr = np.ones(len(sp))
    cls_arr[:len(sps)] = -1

    # select sources not coexist with others in the same pixel 
    mask_inst1, mask_inst2 = mask_insts
    subm_arr, subm0_arr, subx1_arr, subx2_arr, suby1_arr, suby2_arr, \
    subz_arr, subcls_arr = [], [], [], [], [], [], [], []
    for i, (x1, y1, x2, y2) in enumerate(zip(x1_arr, y1_arr, x2_arr, y2_arr)):
        if centnum_map1[int(np.round(x1)), int(np.round(y1))]==1 and \
        centnum_map2[int(np.round(x2)), int(np.round(y2))]==1 and \
        mask_inst1[int(np.round(x1)), int(np.round(y1))]==1 and \
        mask_inst2[int(np.round(x2)), int(np.round(y2))]==1:
            subm_arr.append(m_arr[i])
            subm0_arr.append(m0_arr[i])
            subz_arr.append(z_arr[i])
            subcls_arr.append(cls_arr[i])
            subx1_arr.append(x1)
            suby1_arr.append(y1)
            subx2_arr.append(x2)
            suby2_arr.append(y2)
    subm_arr, subm0_arr, subz_arr, subcls_arr = \
    np.array(subm_arr), np.array(subm0_arr), np.array(subz_arr), np.array(subcls_arr) 
    subx1_arr, suby1_arr, subx2_arr, suby2_arr = \
    np.array(subx1_arr), np.array(suby1_arr), np.array(subx2_arr), np.array(suby2_arr)

    randidx = np.arange(len(subm_arr))
    np.random.shuffle(randidx)
    if inst==1:
        x_arr, y_arr = subx1_arr[randidx], suby1_arr[randidx]
    else:
        x_arr, y_arr = subx2_arr[randidx], suby2_arr[randidx]

    z_arr, m_arr, m0_arr, cls_arr =\
    subz_arr[randidx], subm_arr[randidx], subm0_arr[randidx], subcls_arr[randidx]
    xg_arr, yg_arr, mg_arr, mg0_arr =\
    x_arr[cls_arr==1], y_arr[cls_arr==1], m_arr[cls_arr==1], m0_arr[cls_arr==1]
    zg_arr = z_arr[cls_arr==1]
    xs_arr, ys_arr, ms_arr, ms0_arr =\
    x_arr[cls_arr==-1], y_arr[cls_arr==-1], m_arr[cls_arr==-1], m0_arr[cls_arr==-1]

    srcdat = {}
    srcdat['inst']= inst
    srcdat['ifield'] = ifield
    srcdat['field'] = fieldnamedict[ifield]
    srcdat['sample_type'] = sample_type
    srcdat['m_min'], srcdat['m_max'] = m_min, m_max
    srcdat['Ng'], srcdat['Ns'] = len(xg_arr), len(xs_arr)

    if sample_type == 'all':
        srcdat['xg_arr'], srcdat['yg_arr'] = xg_arr, yg_arr
        srcdat['mg_arr'], srcdat['zg_arr'] = mg_arr, zg_arr
        srcdat['xs_arr'], srcdat['ys_arr'] = xs_arr, ys_arr
        srcdat['ms_arr'] = ms_arr
        srcdat['ms0_arr'] = ms0_arr
        
    elif sample_type == 'jack_random':
        srcdat['Nsub'] = Nsub
        srcdat['sub'] = {}
        for i in range(Nsub):
            srcdat['sub'][i] = {}
            spg = np.arange(i,len(xg_arr),Nsub)
            srcdat['sub'][i]['xg_arr'], srcdat['sub'][i]['yg_arr'] = xg_arr[spg], yg_arr[spg]
            srcdat['sub'][i]['mg_arr'], srcdat['sub'][i]['zg_arr'] = mg_arr[spg], zg_arr[spg]
            srcdat['sub'][i]['mg0_arr'] = mg0_arr[spg]
            sps = np.arange(i,len(xs_arr),Nsub)
            srcdat['sub'][i]['xs_arr'], srcdat['sub'][i]['ys_arr'] = xs_arr[sps], ys_arr[sps]
            srcdat['sub'][i]['ms_arr'] = ms_arr[sps]
            srcdat['sub'][i]['ms0_arr'] = ms0_arr[sps]
            srcdat['sub'][i]['Ng'], srcdat['sub'][i]['Ns'] = len(spg), len(sps)
    
    return srcdat


def run_nonuniform_BG(inst, ifield):
    for im in range(4):
        m_min, m_max = magbindict['m_min'][im],magbindict['m_max'][im]
        stack = stacking(inst, ifield, m_min, m_max, 
            load_from_file=True,run_nonuniform_BG=True)
        
class stacking:
    def __init__(self, inst, ifield, m_min, m_max, srctype='g', 
        savename=None, load_from_file=False, run_nonuniform_BG=False):
        self.inst = inst
        self.ifield = ifield
        self.field = fieldnamedict[ifield]
        self.m_min = m_min
        self.m_max = m_max
        
        if savename is None:
            savename = './stack_data/stackdat_TM%d_%s_%d_%d'%(inst, self.field, m_min, m_max)
        self.savename = savename
        
        if load_from_file:
            stackdat = np.load(savename + '.npy' ,allow_pickle='TRUE').item()
            self.stackdat = stackdat
            if run_nonuniform_BG:
                self.stack_BG(Nbg=64, uniform=False)
                np.save(savename, stackdat)

        else:
            stackdat = self.stack_PS()
            self.stackdat = stackdat
            self.stack_BG(Nbg=64)
            np.save(savename, stackdat)
        
        self._post_process()
    
    def _post_process(self):
        self._get_jackknife_profile()
        self._get_covariance()
        self._get_BG_covariance()
        self._get_BGsub()
        self._get_PSF_from_data()
        self._get_PSF_covariance_from_data()
        self._get_ex_covariance()
        self._get_excess()
        
    def stack_PS(self, srctype='g', dx=1200, unmask=True, verbose=True):

        inst = self.inst
        ifield = self.ifield
        m_min, m_max = self.m_min, self.m_max

        cbmap, psmap, strmask, strnum, mask_inst1, mask_inst2 = \
        load_processed_images(return_names=[(inst,ifield,'cbmap'), 
                                            (inst,ifield,'psmap'),
                                           (inst,ifield,'strmask'), 
                                           (inst,ifield,'strnum'),
                                           (1,ifield,'mask_inst'),
                                           (2,ifield,'mask_inst')])
        if inst==1:
            mask_inst = mask_inst1
        else:
            mask_inst = mask_inst2
            
        srcdat = ps_src_select(inst, ifield, m_min, m_max, [mask_inst1, mask_inst2])
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
        
        cbmapstack, psmapstack, maskstack = 0., 0., 0
        start_time = time.time()
        for isub in range(srcdat['Nsub']):
            stackdat['sub'][isub] = {}

            xls = srcdat['sub'][isub]['x' + srctype + '_arr']
            yls = srcdat['sub'][isub]['y' + srctype + '_arr']
            xss = np.round(xls * 10 + 4.5).astype(np.int32)
            yss = np.round(yls * 10 + 4.5).astype(np.int32)
            ms = srcdat['sub'][isub]['m' + srctype + '_arr']
            rs = get_mask_radius_th(ifield, ms) # arcsec

            print('stacking %s %d < m < %d, #%d subsample, %d sources, t = %.2f min'\
              %(fieldnamedict[ifield], m_min, m_max,isub, len(xls), (time.time()-start_time)/60))

            cbmapstacki, psmapstacki, maskstacki = 0., 0., 0
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
            stackdat['sub'][isub]['profcb100'] = np.sum(cbmapstacki[spi]) / np.sum(maskstacki[spi])
            stackdat['sub'][isub]['profps100'] = np.sum(psmapstacki[spi]) / np.sum(maskstacki[spi])
            stackdat['sub'][isub]['profhit100'] = np.sum(maskstacki[spi])

        ### end isub for loop ###
        
        spmap = np.where(maskstack!=0)
        cbmapstack_norm = np.zeros_like(cbmapstack)
        psmapstack_norm = np.zeros_like(psmapstack)
        cbmapstack_norm[spmap] = cbmapstack[spmap]/maskstack[spmap]
        psmapstack_norm[spmap] = psmapstack[spmap]/maskstack[spmap]
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
        
    def _stackihl_PS_cliplim(self, Nsrc=1000):
        inst = self.inst
        ifield = self.ifield
        m_min = self.m_min
        m_max = self.m_max
        cbmap, psmap, strnum, mask_inst1, mask_inst2 = \
        load_processed_images(return_names=[(inst,ifield,'cbmap'), 
                                            (inst,ifield,'psmap'),
                                           (inst,ifield,'strnum'),
                                           (1,ifield,'mask_inst'),
                                           (2,ifield,'mask_inst')])

        srcdat = ps_src_select(inst, ifield, m_min, m_max, [mask_inst1, mask_inst2], sample_type='all')

        x_arr = np.append(srcdat['xg_arr'],srcdat['xs_arr'])
        y_arr = np.append(srcdat['yg_arr'],srcdat['ys_arr'])
        m_arr = np.append(srcdat['mg_arr'],srcdat['ms_arr'])
        if inst==1:
            mask_inst = mask_inst1
        else:
            mask_inst = mask_inst2
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

        # calculate 
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
            profhitsub = self.stackdat['profhitsub'] - self.stackdat['sub'][isub]['profhitsub']
            self.stackdat['jack'][isub]['profcbsub'] = profcbsub/profhitsub
            self.stackdat['jack'][isub]['profpssub'] = profpssub/profhitsub
            self.stackdat['jack'][isub]['profhitsub'] = profhitsub

            profcb100 = self.stackdat['profcb100']*self.stackdat['profhit100'] - \
            self.stackdat['sub'][isub]['profcb100']*self.stackdat['sub'][isub]['profhit100']
            profps100 = self.stackdat['profps100']*self.stackdat['profhit100'] - \
            self.stackdat['sub'][isub]['profps100']*self.stackdat['sub'][isub]['profhit100']
            profhit100 = self.stackdat['profhit100'] - self.stackdat['sub'][isub]['profhit100']
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
    
    def stack_BG(self, srctype='g', dx=120, verbose=True, Nbg=64, uniform=True):

        inst = self.inst
        ifield = self.ifield
        m_min, m_max = self.m_min, self.m_max

        cbmap, psmap, strmask, mask_inst = \
        load_processed_images(return_names=[(inst,ifield,'cbmap'), 
                                            (inst,ifield,'psmap'),
                                           (inst,ifield,'strmask'), 
                                           (inst,ifield,'mask_inst')])
        Nsrc = self.stackdat['Nsrc']
        self.stackdat['BG'] = {}
        self.stackdat['BG']['Nbg'] = Nbg
        
        # start stacking
        Nbins = len(self.stackdat['rbins'])
        Nsubbins = len(self.stackdat['rsubbins'])
        radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx) # subpix unit
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
            print('stacking %s %d < m < %d, #%d BG, %d sources, t = %.2f min'\
              %(fieldnamedict[ifield], m_min, m_max,isub, Nsrc, (time.time()-start_time)/60))
            
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
            self.stackdat['BG'][isub]['profcb100'] = np.sum(cbmapstacki[spi]) / np.sum(maskstacki[spi])
            self.stackdat['BG'][isub]['profps100'] = np.sum(psmapstacki[spi]) / np.sum(maskstacki[spi])
            self.stackdat['BG'][isub]['profhit100'] = np.sum(maskstacki[spi])

        ### end isub for loop ###

        spmap = np.where(maskstack!=0)
        cbmapstack_norm = np.zeros_like(cbmapstack)
        psmapstack_norm = np.zeros_like(psmapstack)
        cbmapstack_norm[spmap] = cbmapstack[spmap]/maskstack[spmap]
        psmapstack_norm[spmap] = psmapstack[spmap]/maskstack[spmap]
        self.stackdat['cbmapstackBG'] = cbmapstack_norm
        self.stackdat['psmapstackBG'] = psmapstack_norm
        self.stackdat['maskstackBG'] = maskstack
        
        self._get_BG_avg()
        return

    def _get_BG_avg(self):
        Nsub = self.stackdat['BG']['Nbg']
        Nbins = len(self.stackdat['rbins'])
        Nsubbins = len(self.stackdat['rsubbins'])
        data_cb, data_ps = np.zeros([Nsub, Nbins]), np.zeros([Nsub, Nbins])
        data_cbsub, data_pssub = np.zeros([Nsub, Nsubbins]), np.zeros([Nsub, Nsubbins])
        data_cb100, data_ps100 = np.zeros(Nsub), np.zeros(Nsub)

        for isub in range(Nsub):
            data_cb[isub,:] = self.stackdat['BG'][isub]['profcb']
            data_ps[isub,:] = self.stackdat['BG'][isub]['profps']
            data_cbsub[isub,:] = self.stackdat['BG'][isub]['profcbsub']
            data_pssub[isub,:] = self.stackdat['BG'][isub]['profpssub']
            data_cb100[isub] = self.stackdat['BG'][isub]['profcb100']
            data_ps100[isub] = self.stackdat['BG'][isub]['profps100']
            
        self.stackdat['BG']['profcb'] = np.mean(data_cb, axis=0)
        self.stackdat['BG']['profps'] = np.mean(data_ps, axis=0)
        self.stackdat['BG']['profcbsub'] = np.mean(data_cbsub, axis=0)
        self.stackdat['BG']['profpssub'] = np.mean(data_pssub, axis=0)
        self.stackdat['BG']['profcb100'] = np.mean(data_cb100, axis=0)
        self.stackdat['BG']['profps100'] = np.mean(data_ps100, axis=0)

    def _get_BG_covariance(self):
        self._get_BG_avg()
        self.stackdat['BGcov'] = {}
        Nsub = self.stackdat['BG']['Nbg']
        Nbins = len(self.stackdat['rbins'])
        Nsubbins = len(self.stackdat['rsubbins'])
        data_cb, data_ps = np.zeros([Nsub, Nbins]), np.zeros([Nsub, Nbins])
        data_cbsub, data_pssub = np.zeros([Nsub, Nsubbins]), np.zeros([Nsub, Nsubbins])
        data_cb100, data_ps100 = np.zeros(Nsub), np.zeros(Nsub)

        for isub in range(Nsub):
            data_cb[isub,:] = self.stackdat['BG'][isub]['profcb']
            data_ps[isub,:] = self.stackdat['BG'][isub]['profps']
            data_cbsub[isub,:] = self.stackdat['BG'][isub]['profcbsub']
            data_pssub[isub,:] = self.stackdat['BG'][isub]['profpssub']
            data_cb100[isub] = self.stackdat['BG'][isub]['profcb100']
            data_ps100[isub] = self.stackdat['BG'][isub]['profps100']

        covcb = np.zeros([Nbins, Nbins])
        covps = np.zeros([Nbins, Nbins])
        for i in range(Nbins):
            for j in range(Nbins):
                datai, dataj = data_cb[:,i], data_cb[:,j]
                covcb[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
                datai, dataj = data_ps[:,i], data_ps[:,j]
                covps[i,j] = np.mean(datai*dataj) - np.mean(datai)*np.mean(dataj)
        self.stackdat['BGcov']['profcb'] = covcb
        self.stackdat['BGcov']['profps'] = covps
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
        self.stackdat['BGcov']['profcbsub'] = covcb
        self.stackdat['BGcov']['profpssub'] = covps
        self.stackdat['BGcov']['profcbsub_rho'] = self._normalize_cov(covcb)
        self.stackdat['BGcov']['profpssub_rho'] = self._normalize_cov(covps)

        covcb = np.mean(data_cb100**2) - np.mean(data_cb100)**2
        covps = np.mean(data_ps100**2) - np.mean(data_ps100)**2
        self.stackdat['BGcov']['profcb100'] = covcb
        self.stackdat['BGcov']['profps100'] = covps

    def _get_BGsub(self):
        self.stackdat['BGsub'] = {}
        self.stackdat['BGsub']['profcb'] = self.stackdat['profcb'] - self.stackdat['BG']['profcb']
        self.stackdat['BGsub']['profps'] = self.stackdat['profps'] - self.stackdat['BG']['profps']
        self.stackdat['BGsub']['profcbsub'] = self.stackdat['profcbsub'] \
        - self.stackdat['BG']['profcbsub']
        self.stackdat['BGsub']['profpssub'] = self.stackdat['profpssub'] \
        - self.stackdat['BG']['profpssub']
        self.stackdat['BGsub']['profcb100'] = self.stackdat['profcb100'] \
        - self.stackdat['BG']['profcb100']
        self.stackdat['BGsub']['profps100'] = self.stackdat['profps100'] \
        - self.stackdat['BG']['profps100']

    def _get_PSF_from_data(self):
        import json
        loaddir = mypaths['alldat']+'TM' + str(self.inst) + '/'
        with open(loaddir + self.field + '_datafit.json') as json_file:
            data_all = json.load(json_file)
        im = self.stackdat['m_min'] - 16
        data = data_all[im]
        
        scalecb = self.stackdat['BGsub']['profcb'][0]
        scaleps = self.stackdat['BGsub']['profps'][0]
        self.stackdat['PSF'] = {}
        self.stackdat['PSF']['profcb'] = np.array(data['profpsfcbfull'])*scalecb
        self.stackdat['PSF']['profps'] = np.array(data['profpsfpsfull'])*scaleps
        self.stackdat['PSF']['profcbsub'] = np.array(data['profpsfcb'])*scalecb
        self.stackdat['PSF']['profpssub'] = np.array(data['profpsfps'])*scaleps
        self.stackdat['PSF']['profcb100'] = np.array(data['profpsfcb100'])*scalecb
        self.stackdat['PSF']['profps100'] = np.array(data['profpsfps100'])*scaleps


    def _get_PSF_covariance_from_data(self):
        import json
        loaddir = mypaths['alldat']+'TM' + str(self.inst) + '/'
        with open(loaddir + self.field + '_datafit.json') as json_file:
            data_all = json.load(json_file)
        im = self.stackdat['m_min'] - 16
        data = data_all[im]
        
        scalecb = self.stackdat['BGsub']['profcb'][0]
        scaleps = self.stackdat['BGsub']['profps'][0]
        self.stackdat['PSFcov'] = {}
        self.stackdat['PSFcov']['profcb'] = np.array(data['covpsfcbfull'])
        self.stackdat['PSFcov']['profps'] = np.array(data['covpsfpsfull'])
        self.stackdat['PSFcov']['profcbsub'] = np.array(data['covpsfcb'])
        self.stackdat['PSFcov']['profpssub'] = np.array(data['covpsfps'])
        self.stackdat['PSFcov']['profcb100'] = np.array(data['covpsfcb100'])
        self.stackdat['PSFcov']['profps100'] = np.array(data['covpsfps100'])
        self.stackdat['PSFcov']['profcb_rho'] = self._normalize_cov(self.stackdat['PSFcov']['profcb'])
        self.stackdat['PSFcov']['profps_rho'] = self._normalize_cov(self.stackdat['PSFcov']['profps'])
        self.stackdat['PSFcov']['profcbsub_rho'] \
        = self._normalize_cov(self.stackdat['PSFcov']['profcbsub'])
        self.stackdat['PSFcov']['profpssub_rho'] \
        = self._normalize_cov(self.stackdat['PSFcov']['profpssub'])


    def _get_ex_covariance(self):
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
        
        self.stackdat['excov']['profcb_rho'] = self._normalize_cov(self.stackdat['excov']['profcb'])
        self.stackdat['excov']['profps_rho'] = self._normalize_cov(self.stackdat['excov']['profps'])
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



def stack_bigpix(inst, ifield, m_min, m_max, srctype='g', dx=120, verbose=False):

    stackdat = {}
    cbmap, psmap, strmask, strnum, mask_inst1, mask_inst2 = \
    load_processed_images(return_names=[(inst,ifield,'cbmap'), 
                                        (inst,ifield,'psmap'),
                                       (inst,ifield,'strmask'), 
                                       (inst,ifield,'strnum'),
                                       (1,ifield,'mask_inst'),
                                       (2,ifield,'mask_inst')])

    srcdat = ps_src_select(inst, ifield, m_min, m_max, [mask_inst1, mask_inst2])

    if inst==1:
        mask_inst = mask_inst1
    else:
        mask_inst = mask_inst2

    dx = 1200
    profile = radial_prof(np.ones([2*dx+1,2*dx+1]), dx, dx)
    rbinedges, rbins = profile['rbinedges'], profile['rbins']
    stackdat['rbins'] = rbins*0.7
    stackdat['rbinedges'] = rbinedges*0.7
    rbins /= 10 # bigpix
    rbinedges /=10 # bigpix

    dx = 120
    radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx)
    cbmapi = cbmap*strmask*mask_inst
    maski = strmask*mask_inst

    cbmapstack, maskstack = 0., 0
    start_time = time.time()
    for isub in range(srcdat['Nsub']):

        if srctype == 'g':
            Nsrc = srcdat['sub'][isub]['Ng']
            x_arr, y_arr = srcdat['sub'][isub]['xg_arr'], srcdat['sub'][isub]['yg_arr']
        elif srctype == 's':
            Nsrc = srcdat['sub'][isub]['Ns']
            x_arr, y_arr = srcdat['sub'][isub]['xs_arr'], srcdat['sub'][isub]['ys_arr']
        elif srctype == 'bg':
            Nsrc = 100
            x_arr, y_arr = np.random.randint(0,1024,100), np.random.randint(0,1024,100)

        stackdat[isub] = {}
        if verbose:
            print('stacking %s %d < m < %d, #%d, %d src, t = %.2f min'\
              %(fieldnamedict[ifield], m_min, m_max, isub, 
                Nsrc, (time.time()-start_time)/60))

        cbmapstacki, maskstacki = 0., 0
        for i in range(Nsrc):
            xi, yi = int(x_arr[i]), int(y_arr[i])
            radmap = make_radius_map(cbmap, xi, yi) # large pix units

            # zero padding
            mcb = np.pad(cbmapi, ((dx,dx),(dx,dx)), 'constant')
            k = np.pad(maski, ((dx,dx),(dx,dx)), 'constant')
            xi += dx
            yi += dx

            # cut stamp
            cbmapstamp = mcb[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]
            maskstamp = k[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]

            cbmapstacki += cbmapstamp
            maskstacki += maskstamp

        ### end source for loop ###
        cbmapstack += cbmapstacki
        maskstack += maskstacki

        Nbins = len(rbins)
        profcb_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
        for ibin in range(Nbins):
            spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                           (radmapstamp<rbinedges[ibin+1]))
            profcb_arr[ibin] += np.sum(cbmapstacki[spi])
            hit_arr[ibin] += np.sum(maskstacki[spi])
        stackdat[isub]['profcb'] = profcb_arr/hit_arr
        stackdat[isub]['profhit'] = hit_arr

    profcb_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
    for ibin in range(Nbins):
        spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                       (radmapstamp<rbinedges[ibin+1]))
        profcb_arr[ibin] += np.sum(cbmapstack[spi])
        hit_arr[ibin] += np.sum(maskstack[spi])
    spbin = np.where(hit_arr!=0)[0]
    profcb_norm = np.zeros_like(profcb_arr)
    profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
    stackdat['profcb'] = profcb_norm
    stackdat['profhit'] = hit_arr

    data_cb = np.zeros([srcdat['Nsub'], Nbins])
    stackdat['jack'] = {}
    for isub in range(srcdat['Nsub']):
        stackdat['jack'][isub] = {}
        profcb = stackdat['profcb']*stackdat['profhit'] - \
        stackdat[isub]['profcb']*stackdat[isub]['profhit']
        profhit = stackdat['profhit'] - stackdat[isub]['profhit']
        stackdat['jack'][isub]['profcb'] = profcb/profhit
        stackdat['jack'][isub]['profhit'] = profhit
        data_cb[isub,:] = profcb/profhit

    stackdat['profcb_err'] = np.sqrt(np.var(data_cb, axis=0)*srcdat['Nsub'])

    return stackdat

class stacking_mock:
    def __init__(self, inst, m_min, m_max, srctype='g', 
                 catname = 'PanSTARRS', ifield = 8, pixsize=7):
        self.inst = inst
        self.m_min = m_min
        self.m_max = m_max
        self.pixsize = pixsize
        
        if catname=='PanSTARRS':
            self.catname = catname
            self.catdir = mypaths['PScatdat']
            if ifield in [4,5,6,7,8]:
                self.ifield = ifield
                self.field = fieldnamedict[ifield]
            else:
                raise Exception('ifield invalid (must be int between 4-8)')
        elif catname=='HSC':
            self.catname = catname
            self.catdir = mypaths['HSCcatdat']
            if ifield in range(12):
                self.ifield = ifield
                self.field = 'W05_%d'%ifield
            else:
                raise Exception('ifield invalid (must be int between 0-11)')
        else:
            raise Exception('catname invalid (enter PanSTARRS or HSC)')
        
        if srctype in [None, 's', 'g', 'u']:
            self.srctype = srctype
        else:
            raise Exception('srctype invalid (must be None, "s", "g", or "u")')
        
        self.Npix_cb = 1024
        self.Nsub = 10
        
        self.xls, self.yls, self.ms, self.xss, self.yss, self.ms_inband = self._load_catalog()
        
        self._get_mask_radius()
        
    def _load_catalog(self):
        Npix_cb = self.Npix_cb
        Nsub = self.Nsub
        
        df = pd.read_csv(self.catdir + self.field + '.csv')
        if self.catname == 'PanSTARRS':
            xls = df['y'+str(self.inst)].values
            yls = df['x'+str(self.inst)].values
            ms = df['I_comb'].values
            clss = df['sdssClass'].values
            if self.srctype == 's':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==6))[0]
            elif self.srctype == 'g':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==3))[0]
            elif self.srctype == 'u':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==-99))[0]
            elif self.srctype is None:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max))[0]            

        else:
            xls = df['x'].values - 2.5
            yls = df['y'].values - 2.5
            ms = df['Imag'].values
            clss = df['cls'].values
            if self.srctype == 's':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==-1))[0]
            elif self.srctype == 'g':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==1))[0]
            elif self.srctype == 'u':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==2))[0]
            elif self.srctype is None:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max))[0]            
            
        xls = xls[sp]
        yls = yls[sp]
        clss = clss[sp]
        ms = ms[sp]
        ms_inband = ms.copy()
        if self.inst == 2:
            if self.catname == 'PanSTARRS':
                ms_inband = df['H_comb'].values[sp]
            else:
                ms_inband = df['Hmag'].values[sp]
            
        sp = np.where((xls > 0) & (yls > 0) \
                      & (xls < Npix_cb) & (yls < Npix_cb))[0]
        xls = xls[sp]
        yls = yls[sp]
        ms = ms[sp]
        ms_inband = ms_inband[sp]

        xss = np.round(xls * Nsub + (Nsub/2 - 0.5)).astype(np.int32)
        yss = np.round(yls * Nsub + (Nsub/2 - 0.5)).astype(np.int32)

        return xls, yls, ms, xss, yss, ms_inband
    
    def _get_mask_radius(self, mask_func = MZ14_mask):
        rs =  mask_func(self.inst, self.xls, self.yls, self.ms_inband, return_radius=True)  
        self.rs = rs
        
    def _image_finegrid(self, image):
        Nsub = self.Nsub
        w, h  = np.shape(image)
        image_new = np.zeros([w*Nsub, h*Nsub])
        for i in range(Nsub):
            for j in range(Nsub):
                image_new[i::Nsub, j::Nsub] = image
        return image_new

    def run_stacking(self, mapin, mask, num, mask_inst=None, dx=1200, verbose=True):
        
        self.dx = dx
        Nsub = self.Nsub
        
        self.xss = np.round(self.xls * Nsub + (Nsub/2 - 0.5)).astype(np.int32)
        self.yss = np.round(self.yls * Nsub + (Nsub/2 - 0.5)).astype(np.int32)
        self._get_mask_radius()
        
        mask_inst = np.ones_like(mapin) if mask_inst is None else mask_inst
        
        mapstack = 0.
        maskstack = 0
        for i,(xl,yl,xs,ys,r) in enumerate(zip(self.xls,self.yls,self.xss,self.yss,self.rs)):
            if len(self.xls)>20:
                if verbose and i%(len(self.xls)//20)==0:
                    print('stacking %d / %d (%.1f %%)'\
                          %(i, len(self.xls), i/len(self.xls)*100))
            
            # unmask source
            radmap = make_radius_map(mapin, xl, yl)
            maski = mask.copy()
            maski[(radmap < r / self.pixsize) & (num==1) & (mask_inst==1)] = 1
            m = mapin * maski
            m = self._image_finegrid(m)
            k = self._image_finegrid(maski)
            
            # zero padding
            m = np.pad(m, ((dx,dx),(dx,dx)), 'constant')
            k = np.pad(k, ((dx,dx),(dx,dx)), 'constant')
            xs += dx
            ys += dx
            
            # cut stamp
            mapstamp = m[xs - dx: xs + dx + 1, ys - dx: ys + dx + 1]
            maskstamp = k[xs - dx: xs + dx + 1, ys - dx: ys + dx + 1]
            
            mapstack += mapstamp
            maskstack += maskstamp
            
        stack = np.zeros_like(mapstack)
        sp = np.where(maskstack!=0)
        stack[sp] = mapstack[sp] / maskstack[sp]
        stack[maskstack==0] = 0
            
        return stack, maskstack, mapstack
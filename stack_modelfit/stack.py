from mask import *
import pandas as pd
import time

def load_processed_images(return_names=[(1,4,'cbmap'), (1,4,'psmap')]):
    '''
    get the images processed by stack_preprocess.m
    
    Input:
    =======
    return_names: list of items (inst, ifield, map name)
    
    Ouput:
    =======
    return_maps: list of map of the input return_names
    
    '''
    img_names = {'rawmap':0, 'rawmask':1, 'DCsubmap':2, 'FF':3, 'FFunholy':4,
                'map':5, 'cbmap':6, 'psmap':7, 'mask_inst':8, 'strmask':9, 'strnum':10}
    data = {}
    data[1] = loadmat(mypaths['alldat'] + 'TM' + str(1) + '/stackmapdatarr.mat')['data']
    data[2] = loadmat(mypaths['alldat'] + 'TM' + str(2) + '/stackmapdatarr.mat')['data']
    
    return_maps = []
    for inst,ifield,name in return_names:
        mapi = data[inst][ifield-4][img_names[name]]
        return_maps.append(mapi)
    return return_maps


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

class stacking:
    def __init__(self, inst, ifield, m_min, m_max, srctype='g', savename=None, load_from_file=False):
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
        else:
            stackdat = self.stack_PS()        
            np.save(savename, stackdat)
        
        self.stackdat = stackdat
        

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
        stackdat['inst']= inst
        stackdat['ifield'] = ifield
        stackdat['field'] = fieldnamedict[ifield]
        stackdat['m_min'], stackdat['m_max'] = m_min, m_max
        stackdat['Nsrc'] = srcdat['N' + srctype]
        stackdat['Nsub'] = srcdat['Nsub']
        stackdat['sub'] = {}

        # start stacking
        Nbins = len(cliplim['rbins'])
        radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx) # subpix unit
        rbinedges = stackdat['rbinedges']/0.7 # subpix unit

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

            profcb_arr, profps_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
            for ibin in range(Nbins):
                spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                               (radmapstamp<rbinedges[ibin+1]))
                profcb_arr[ibin] += np.sum(cbmapstacki[spi])
                profps_arr[ibin] += np.sum(psmapstacki[spi])
                hit_arr[ibin] += np.sum(maskstacki[spi])

            cbmapstack += cbmapstacki
            psmapstack += psmapstacki
            maskstack += maskstacki

            spbin = np.where(hit_arr!=0)[0]
            profcb_norm, profps_norm = np.zeros_like(profcb_arr), np.zeros_like(profps_arr)
            profcb_norm[spbin] = profcb_arr[spbin]/hit_arr[spbin]
            profps_norm[spbin] = profps_arr[spbin]/hit_arr[spbin]

            stackdat['sub'][isub]['profcb'] = profcb_norm
            stackdat['sub'][isub]['profps'] = profps_norm
            stackdat['sub'][isub]['profhit'] = hit_arr

        spmap = np.where(maskstack!=0)
        cbmapstack_norm = np.zeros_like(cbmapstack)
        psmapstack_norm = np.zeros_like(psmapstack)
        cbmapstack_norm[spmap] = cbmapstack[spmap]/maskstack[spmap]
        psmapstack_norm[spmap] = psmapstack[spmap]/maskstack[spmap]
        stackdat['cbmapstack'] = cbmapstack_norm
        stackdat['psmapstack'] = psmapstack_norm
        stackdat['maskstack'] = maskstack

        profcb_arr, profps_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
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
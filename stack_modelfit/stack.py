from mask import *
import pandas as pd

class stacking:
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
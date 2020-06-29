import os
from scipy.io import loadmat
import numpy as np
from ciber_info import *
from mask import *
from srcmap import *
from skimage import restoration
from scipy.optimize import curve_fit

class get_frame_data():
    
    def __init__(self):
        return
    
    def get_frames(self, inst, ifield, skipfr=2):
        field = fieldnamedict[ifield]
        listing = self._get_frame_fnames(inst, field)
        frames = []
        for fname in listing:
            frames.append(loadmat(fname)['arraymap'])
        frames = np.array(frames)
        frames = frames[skipfr:,...]
        frames = frames.astype(float)
        return frames
    
    def _get_frame_fnames(self, inst, field):
        timedown, timeup = self._get_field_times(field)
        listing = []
        timelist = []
        for fname in os.listdir(mypaths['framedir']):
            t = float(fname[4:-4])
            if t > timedown and t < timeup and int(fname[2]) == inst:
                timelist.append(t)
        timelist = np.sort(timelist)
        for t in timelist:
            fname = mypaths['framedir'] + 'TM' + str(inst) + '_' + '%.5f'%t + '.mat'
            listing.append(fname)
        
        return listing

    def _get_field_times(self, field):
        '''
        The flight frame data start and end time label, this info is from
        ciber_analysis/Matlab/util/internal/get_field_times.m
        '''
        t0 = 11100.75000

        if field == 'DGL':
            timedown = t0 + 148
            timeup = t0 + 220 

        elif field == 'NEP':
            timedown = t0 + 233
            timeup = t0 + 299

        elif field == 'Lockman':
            timedown = t0 + 316
            timeup = t0 + 370

        elif field == 'elat10':
            timedown = t0 + 387
            timeup = t0 + 436

        elif field == 'elat30':
            timedown = t0 + 450
            timeup = t0 + 500

        elif field == 'BootesB':
            timedown = t0 + 513
            timeup = t0 + 569

        elif field == 'BootesA':
            timedown = t0 + 581
            timeup = t0 + 636

        elif field == 'SWIRE':
            timedown = t0 + 655
            timeup = t0 + 705

        return timedown, timeup
    
class image_reduction:
    def __init__(self, inst):
        self.inst = inst
        self.DCtemplate, self.mask_inst = self.get_DC_mask_inst(self.inst)
        self.ts_process() 
        self.get_psf(self.inst, self.stackmapdat)    
        self.get_strmask(self.inst)
        self.get_srcmap(self.inst)
        self.FF_correction(self.inst)
        self.get_mask_inst(self.inst)
        
    def ts_process(self):
        stackmapdat = {}
        for ifield in [4,5,6,7,8]:
            stackmapdat[ifield] = {}
            tsmask = self.get_tsmask(self.inst, ifield)
            linmap, negmask, long, Nfr = self.linearized_map(self.inst, ifield)
            mask = self.mask_inst * tsmask * negmask
            
            stackmapdat[ifield]['Nfr'] = Nfr
            stackmapdat[ifield]['rawmap'] = long
            stackmapdat[ifield]['rawmask'] = mask
            stackmapdat[ifield]['DCsubmap'] = linmap - self.DCtemplate
        
        self.stackmapdat = stackmapdat
    
    def get_psf(self, inst, stackmapdat):
        '''
        fname = mypaths['alldat'] + 'TM'+ str(inst) + '/psfdata.pkl'
        if os.path.exists(fname):
            with open(fname,"rb") as f:
                psfdata = pickle.load(f)
        else:
            ###import this before running this block
            ### from psfstack import *
            stack_psf(inst, self.stackmapdat)
            with open(fname,"rb") as f:
                psfdata = pickle.load(f)

        def beta_function(r, beta, rc, norm):
            return norm * (1 + (r / rc)**2)**(-3.*beta/2)

        pix_map = self._pix_func_substack()

        for ifield in [4,5,6,7,8]:
            rbins = psfdata[ifield]['rbins']
            psf_map = restoration.richardson_lucy\
            (psfdata[ifield]['stackmap']/np.sum(psfdata[ifield]['stackmap']),
             pix_map, 10)
            psfprof = radial_prof(psf_map)['prof']
            
            (beta, rc, norm), _ = curve_fit(beta_function, rbins[rbins < 30]
                                ,psfprof[rbins < 30]/psfprof[0],
                                sigma=psfprof[rbins < 30])
            
            dx = 1200
            radmap = make_radius_map(np.zeros([2*dx+1, 2*dx+1]),dx, dx)*0.7
            psf_map_beta = beta_function(radmap, beta, rc, norm)
            norm = norm / np.sum(psf_map_beta)

            self.stackmapdat[ifield]['PSFparams'] = (beta, rc, norm)
        '''

        # the best fit params is written already written here
        for ifield in [4,5,6,7,8]:
            beta, rc, norm = PSF_model_dict[self.inst][ifield]
            self.stackmapdat[ifield]['PSFparams'] = (beta, rc, norm)

    def get_strmask(self, inst):

        fname = mypaths['alldat'] + 'TM'+ str(inst) + '/strmaskdat'
        try:
            strmaskdat = np.load(fname + '.npy')
        except OSError as error:
            strmasks, strnums = [], []

            for ifield in [4,5,6,7,8]:
                strmask, strnum = Ith_mask(inst, ifield)
                strmasks.append(strmask)
                strnums.append(strnum)

            strmaskdat = np.stack((np.array(strmasks), np.array(strnums)))
            np.save(fname,strmaskdat)

        for i,ifield in enumerate([4,5,6,7,8]):
            strmask, strnum = strmaskdat[0,i,...], strmaskdat[1,i,...]
            self.stackmapdat[ifield]['strmask'] = strmask
            self.stackmapdat[ifield]['strnum'] = strnum
            
    def get_srcmap(self, inst):

        fname = mypaths['alldat'] + 'TM'+ str(inst) + '/srcmapdat'
        try:
            srcmapdat = np.load(fname + '.npy')
        except OSError as error:
            srcmaps = []

            for ifield in [4,5,6,7,8]:
                print('make srcmap in %s'%fieldnamedict[ifield])
                make_srcmap_class = make_srcmap(inst, srctype=None, catname='2MASS',
                                ifield=ifield, psf_ifield=ifield)
                srcmap2m = make_srcmap_class.run_srcmap(ptsrc=True)
                make_srcmap_class = make_srcmap(inst, srctype=None, catname='PanSTARRS',
                                                ifield=ifield, psf_ifield=ifield)
                srcmapps = make_srcmap_class.run_srcmap(ptsrc=True)
                
                srcmaps.append(srcmap2m + srcmapps)

            srcmapdat = np.array(srcmaps)
            np.save(fname,srcmapdat)
        for i,ifield in enumerate([4,5,6,7,8]):
            self.stackmapdat[ifield]['srcmap'] = srcmapdat[i,...]

    def get_DC_mask_inst(self, inst):
        inst = self.inst
        DCdir = mypaths['DCdir']
        DCtemplate = loadmat(DCdir + 'band' + str(inst) + '_DCtemplate.mat')['DCtemplate']
        mask_inst = loadmat(DCdir + 'band' + str(inst) + '_mask_inst.mat')['mask_inst']
        return DCtemplate, mask_inst
    
    def get_tsmask(self, inst, ifield):
        '''
        Mask bad time-stream pixels
        '''
        frames = get_frame_data().get_frames(inst, ifield)

        Nfr = frames.shape[0]
        dframes = frames[1:] - frames[:-1]

        # mask the pixel there is ts diff > 5e4
        diff_clip_mask = np.zeros_like(dframes)
        diff_clip_mask[np.where(abs(dframes) > 5e4)] = 1
        diff_clip_mask = np.sum(diff_clip_mask, axis=0)

        # mask the pixel if there is ts diff > 100 * median ts diff
        diff_median = np.median(abs(dframes), axis=0)
        diff_median_stack = np.repeat(diff_median[np.newaxis,...], Nfr-1, axis=0)
        med_clip_mask = np.zeros_like(dframes)
        med_clip_mask[np.where(abs(dframes) > 100*diff_median_stack)] = 1
        med_clip_mask = np.sum(med_clip_mask, axis=0)
        med_clip_mask[diff_median==0] = 1

        tsmask = np.ones_like(med_clip_mask)
        tsmask[diff_clip_mask!=0] = 0
        tsmask[med_clip_mask!=0] = 0

        return tsmask

    def _frame_linefit(self, frames, return_offset=False):

        Nfr,Npix = frames.shape[:2]
        poly = np.polynomial.polynomial.polyfit(np.arange(Nfr), frames.reshape(Nfr,-1), 1)
        poly = poly.reshape(2,Npix,Npix)

        if return_offset:
            return poly[1,...], poly[0,...]

        return poly[1,...]

    def linearized_map(self, inst, ifield):
        frames = get_frame_data().get_frames(inst, ifield)

        th = -5000
        if ifield == 5:
            frames = frames[-10:]
            th = -2000

        Nfr,Npix = frames.shape[:2]
        long, off = self._frame_linefit(frames, return_offset=True)
        short = self._frame_linefit(frames[:4])

        # mask negative pixel
        negmask = np.ones((Npix, Npix))
        negmask[long>0] = 0

        # linearize
        linmap = long.copy()
        linmap[(frames[-1] - off) < th] = short[(frames[-1] - off) < th]

        return linmap, negmask, long, Nfr
    
    def FF_correction(self, inst):
        for i in [4,5,6,7,8]:
            FFpix, stack_mask = np.zeros_like(self.DCtemplate), np.zeros_like(self.DCtemplate)

            for j in [4,5,6,7,8]:
                if j==i:
                    continue
                mapin = -self.stackmapdat[j]['DCsubmap'].copy()
                strmask = self.stackmapdat[j]['strmask'].copy()
                mask0 = self.stackmapdat[j]['rawmask'].copy()
                mask = sigma_clip_mask(mapin, strmask*mask0, iter_clip=3, sig=5)
                FFpix += mapin * mask / np.sqrt(np.mean(mapin[mask==1]))
                stack_mask += mask * np.sqrt(np.mean(mapin[mask==1]))

            sp = np.where(stack_mask!=0)
            FFpix[sp] = FFpix[sp] / stack_mask[sp]
            stack_mask[stack_mask>0] = 1
            FFsm = image_smooth_gauss(FFpix, mask=stack_mask, 
                stddev=3, return_unmasked=True)
            FFsm[FFsm!=FFsm] = 0

            FF = FFpix.copy()
            spholes = np.where((self.stackmapdat[i]['rawmask']==1) & (stack_mask==0))
            FF[spholes] = FFsm[spholes]
            self.stackmapdat[i]['FFpix'] = FFpix
            self.stackmapdat[i]['FFsm'] = FFsm
            self.stackmapdat[i]['FF'] = FF

            FFcorrmap = self.stackmapdat[i]['DCsubmap'].copy()
            FFcorrmap[FF!=0] = FFcorrmap[FF!=0] / FF[FF!=0]
            FFcorrmap[self.stackmapdat[i]['rawmask']==0]=0
            FFcorrmap[FF==0]=0
            self.stackmapdat[i]['map'] = FFcorrmap
    
    def get_mask_inst(self, inst):

        for ifield in [4,5,6,7,8]:
            mask_inst = self.stackmapdat[ifield]['rawmask'].copy()
            mask_inst *= self.crmask(inst, ifield)
            strmask = self.stackmapdat[ifield]['strmask'].copy()
            mapin = -self.stackmapdat[ifield]['map'].copy()

            # clip image
            sigmask = strmask * mask_inst
            Q1 = np.percentile(mapin[(sigmask==1)], 25)
            Q3 = np.percentile(mapin[(sigmask==1)], 75)
            clipmin = Q1 - 3 * (Q3 - Q1)
            clipmax = Q3 + 3 * (Q3 - Q1)
            sigmask[(mapin > clipmax)] = 0
            sigmask[(mapin < clipmin)] = 0
            mask_inst[sigmask!=strmask*mask_inst] = 0

            sigmask0 = sigmask
            # clip residual point sources
            mapin_sm = image_smooth_gauss(mapin, mask=sigmask, stddev=1)
            sigmask = strmask * mask_inst
            Q1 = np.percentile(mapin_sm[(sigmask==1)], 25)
            Q3 = np.percentile(mapin_sm[(sigmask==1)], 75)
            clipmin = Q1 - 3 * (Q3 - Q1)
            clipmax = Q3 + 3 * (Q3 - Q1)
            sigmask[(mapin_sm > clipmax)] = 0
            sigmask[(mapin_sm < clipmin)] = 0
            mask_inst[sigmask!=strmask*mask_inst] = 0

            self.stackmapdat[ifield]['mask_inst'] = mask_inst   

    
    def crmask(self, inst, ifield):
        '''
        Bad region identified by eye. Possibly hitted by cosmic rays
        '''
        crmask = np.ones_like(self.DCtemplate)
        size = crmask.shape
        if (inst, ifield) == (1, 8):
            crmask1 = self._circular_mask(39, 441, 40, size)
            crmask2 = self._circular_mask(341, 741, 10, size)
            crmask = crmask * crmask1 * crmask2
        
        elif (inst, ifield) == (2, 8):
            crmask1 = self._circular_mask(394, 439, 10, size)
            crmask2 = self._circular_mask(799, -1, 60, size)
            crmask = crmask * crmask1 * crmask2
        
        elif (inst, ifield) == (2,5):
            crmask1 = self._elliptical_mask(219, 214, 15, 60, 80, size)
            crmask2 = self._elliptical_mask(89, 709, 20, 80, 60, size)
            crmask = crmask * crmask1 * crmask2
        
        return crmask
    
    def _circular_mask(self, x0, y0, r, size):
        mask = np.ones(size)
        radmap = make_radius_map(mask, x0, y0)
        mask[radmap<r] = 0
        return mask

    def _elliptical_mask(self, x0, y0, a, b, theta, size):
        mask = np.ones(size)
        xx, yy = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
        th = theta*np.pi/180
        xxnew = xx*np.cos(th) - yy*np.sin(th)
        yynew = xx*np.sin(th) + yy*np.cos(th)

        xnew = x0*np.cos(th) - y0*np.sin(th)
        ynew = x0*np.sin(th) + y0*np.cos(th)

        radmap = ((xxnew - xnew) / a)**2 + ((yynew - ynew) / b)**2
        mask[radmap < 1] = 0

        return mask
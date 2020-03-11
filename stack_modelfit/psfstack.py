from stack_ancillary import *

def stack_psf(inst, stackmapdat, m_min=12, m_max=14, Nsub=10, 
    Nsub_single=True, savedata=True):
    print('Stack 2MASS stars for PSF ...')

    # get data & mask
    psfdat = {}
    for ifield in [4,5,6,7,8]:
        print('get data & mask in %s'%fieldnamedict[ifield])

        psfdat[ifield] = {}

        DCsubmap = stackmapdat[ifield]['DCsubmap'].copy()
        rawmask = stackmapdat[ifield]['rawmask'].copy()

        psfdat[ifield]['DCsubmap'] = DCsubmap
        psfdat[ifield]['mask_inst'] = rawmask

        catdir = mypaths['2Mcatdat']
        df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')

        xs = df['y'+str(inst)].values
        ys = df['x'+str(inst)].values
        ms = df['j'].values + 2.5*np.log10(1594./3631.) # convert to vega mag
        sp = np.where((ms < 17) & (xs>-20) & (xs<1044) & (ys>-20) & (ys<1044))[0]
        xs, ys, ms = xs[sp], ys[sp], ms[sp]
        rs = -6.25 * ms + 110

        strmask = np.ones([1024,1024])
        strnum = np.zeros([1024,1024])
        for i,(x,y,r) in enumerate(zip(xs, ys, rs)):
            radmap = make_radius_map(strmask, x, y)
            strmask[radmap < r/7.] = 0
            strnum[radmap < r/7.] += 1

        mask_inst = rawmask.copy()
        Q1 = np.percentile(DCsubmap[(mask_inst*strmask==1)], 25)
        Q3 = np.percentile(DCsubmap[(mask_inst*strmask==1)], 75)
        clipmin = Q1 - 3 * (Q3 - Q1)
        clipmax = Q3 + 3 * (Q3 - Q1)
        mask_inst[(DCsubmap > clipmax) & (mask_inst*strmask==1)] = 0
        mask_inst[(DCsubmap < clipmin) & (mask_inst*strmask==1)] = 0

        psfdat[ifield]['mask_inst'] = mask_inst
        psfdat[ifield]['strmask'] = strmask
        psfdat[ifield]['strnum'] = strnum

    # get FF
    for i in [4,5,6,7,8]:
        FFpix, stack_mask = np.zeros_like(strmask), np.zeros_like(strmask)

        for j in [4,5,6,7,8]:
            if j==i:
                continue
            mapin = -psfdat[j]['DCsubmap'].copy()
            strmask = psfdat[j]['strmask'].copy()
            mask_inst = psfdat[j]['mask_inst'].copy()
            mask = mask_inst*strmask
            FFpix += mapin * mask / np.sqrt(np.mean(mapin[mask==1]))
            stack_mask += mask * np.sqrt(np.mean(mapin[mask==1]))

        sp = np.where(stack_mask!=0)
        FFpix[sp] = FFpix[sp] / stack_mask[sp]
        stack_mask[stack_mask>0] = 1
        FFsm = image_smooth_gauss(FFpix, mask=stack_mask, 
            stddev=3, return_unmasked=True)
        FFsm[FFsm!=FFsm] = 0

        FF = FFpix.copy()
        spholes = np.where((psfdat[i]['mask_inst']==1) & (stack_mask==0))
        FF[spholes] = FFsm[spholes]

        FFcorrmap = psfdat[i]['DCsubmap'].copy()
        FFcorrmap[FF!=0] = FFcorrmap[FF!=0] / FF[FF!=0]
        FFcorrmap[psfdat[i]['mask_inst']==0]=0
        FFcorrmap[FF==0]=0

        psfdat[i]['map'] = FFcorrmap

        mask = mask_inst.copy()
        strmask = psfdat[i]['strmask'].copy()
        Q1 = np.percentile(-FFcorrmap[(mask*strmask==1)], 25)
        Q3 = np.percentile(-FFcorrmap[(mask*strmask==1)], 75)
        clipmin = Q1 - 3 * (Q3 - Q1)
        clipmax = Q3 + 3 * (Q3 - Q1)
        mask[(-FFcorrmap > clipmax) & (mask*strmask==1)] = 0
        mask[(-FFcorrmap < clipmin) & (mask*strmask==1)] = 0

        psfdat[i]['mask'] = mask
        
    stack_class = stacking_mock(inst)
    psfdata = {}
    for ifield in [4,5,6,7,8]:
        print('stack %s'%fieldnamedict[ifield])

        mapin = -psfdat[ifield]['map'].copy()
        mask_inst = psfdat[ifield]['mask'].copy()
        strmask = psfdat[ifield]['strmask'].copy()
        strnum = psfdat[ifield]['strnum'].copy()
        mapin = mapin - np.mean(mapin[mask_inst*strmask==1])

        catdir = mypaths['2Mcatdat']
        df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')

        xs = df['y'+str(inst)].values
        ys = df['x'+str(inst)].values
        ms = df['j'].values + 2.5*np.log10(1594./3631.) # convert to vega mag
        sp = np.where((ms>m_min) & (ms<m_max) &\
         (xs>-0.5) & (xs<1023.5) & (ys>-0.5) & (ys<1023.5))[0]
        xs, ys, ms = xs[sp], ys[sp], ms[sp]
        rs = -6.25 * ms + 110

        profs = []
        mapstack, maskstack = 0., 0.

        if Nsub_single:
            Nsub = len(xs)

        for isub in range(Nsub):
            print('stack PSF %s %d/%d'%(fieldnamedict[ifield],isub,Nsub))
            stack_class.xls = xs[isub::Nsub]
            stack_class.yls = ys[isub::Nsub]
            stack_class.ms = ms[isub::Nsub]
            stack_class.rs = rs[isub::Nsub]

            stackdat, stacki, maskstacki, mapstacki \
            = stack_class.run_stacking(mapin, mask_inst*strmask, strnum, 
                                       mask_inst=mask_inst,return_all=True,
                                        update_mask=False, verbose=True)
            mapstack += mapstacki
            maskstack += maskstacki

            profs.append(stackdat['prof'])

        profs = np.array(profs)
        prof_err = np.std(profs, axis=0) / np.sqrt(Nsub) 

        stack = np.zeros_like(mapstack)
        sp = np.where(maskstack!=0)
        stack[sp] = mapstack[sp] / maskstack[sp]
        stack[maskstack==0] = 0

        rbins, rbinedges, dx = stackdat['rbins']/0.7, stackdat['rbinedges']/0.7, stack_class.dx
        Nbins = len(rbins)
        radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx)
        prof_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
        for ibin in range(Nbins):
            spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                           (radmapstamp<rbinedges[ibin+1]))
            prof_arr[ibin] += np.sum(mapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        prof_norm = np.zeros_like(prof_arr)
        prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]

        psfdata[ifield] = {}
        psfdata[ifield]['Nsrc'] = len(xs)
        psfdata[ifield]['rbins'] = stackdat['rbins'].copy()
        psfdata[ifield]['rbinedges'] = stackdat['rbinedges'].copy()
        psfdata[ifield]['prof'] = prof_norm
        psfdata[ifield]['profhit'] = hit_arr
        psfdata[ifield]['prof_err'] = prof_err
        psfdata[ifield]['stackmap'] = stack

    if savedata:
        fname = mypaths['alldat'] + 'TM'+ str(inst) + '/psfdata.pkl'
        with open(fname, "wb") as f:
            pickle.dump(psfdata, f)
            
        return

    return psfdata
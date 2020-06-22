from stack_ancillary import *

def stack_psf(inst, stackmapdat, m_min=12, m_max=14, Nsub=10,
    ifield_arr=[4,5,6,7,8], Nsub_single=True, savedata=True,
     save_stackmap=True, catname='2MASS'):
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

        if catname == '2MASS':
            catdir = mypaths['2Mcatdat']
            df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')
            xs = df['y'+str(inst)].values
            ys = df['x'+str(inst)].values
            ms = df['j'].values + 2.5*np.log10(1594./3631.) # convert to vega mag
            sp = np.where((ms < 17) & (xs>-20) & (xs<1044) & (ys>-20) & (ys<1044))[0]
            xs, ys, ms = xs[sp], ys[sp], ms[sp]
            rs = -6.25 * ms + 110
        elif catname == 'GAIA':
            catdir = mypaths['GAIAcatdat']
            df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')
            xs = df['y'+str(inst)].values
            ys = df['x'+str(inst)].values
            ms = df['phot_g_mean_mag'].values
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
    
    # stack
    stack_class = stacking_mock(inst)
    psfdata = {}
    for ifield in ifield_arr:
        print('stack %s'%fieldnamedict[ifield])

        mapin = -psfdat[ifield]['map'].copy()
        mask_inst = psfdat[ifield]['mask'].copy()
        strmask = psfdat[ifield]['strmask'].copy()
        strnum = psfdat[ifield]['strnum'].copy()
        mapin = mapin - np.mean(mapin[mask_inst*strmask==1])

        if catname == '2MASS':
            catdir = mypaths['2Mcatdat']
            df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')
            xs = df['y'+str(inst)].values
            ys = df['x'+str(inst)].values
            ms = df['j'].values + 2.5*np.log10(1594./3631.) # convert to vega mag
            sp = np.where((ms>m_min) & (ms<m_max) &\
             (xs>-0.5) & (xs<1023.5) & (ys>-0.5) & (ys<1023.5))[0]
            xs, ys, ms = xs[sp], ys[sp], ms[sp]
            rs = -6.25 * ms + 110
        elif catname == 'GAIA':
            catdir = mypaths['GAIAcatdat']
            df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')
            df = df[df['parallax']==df['parallax']]
            xs = df['y'+str(inst)].values
            ys = df['x'+str(inst)].values
            ms = df['phot_g_mean_mag'].values
            parallax = df['parallax'].values
            sp = np.where((ms>m_min) & (ms<m_max) &\
             (xs>-0.5) & (xs<1023.5) & (ys>-0.5) & (ys<1023.5) &\
              (parallax > 1/5e3))[0]
            xs, ys, ms = xs[sp], ys[sp], ms[sp]
            rs = -6.25 * ms + 110

        prof_arr = []
        profhit_arr = []
        profsub_arr = []
        profsubhit_arr = []
        mapstack, maskstack = 0., 0.

        if len(xs) < Nsub:
            Nsub_single = True

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

            prof_arr.append(stackdat['prof'])
            profhit_arr.append(stackdat['profhit'])
            profsub_arr.append(stackdat['profsub'])
            profsubhit_arr.append(stackdat['profhitsub'])

        stack = np.zeros_like(mapstack)
        sp = np.where(maskstack!=0)
        stack[sp] = mapstack[sp] / maskstack[sp]
        stack[maskstack==0] = 0

        prof_arr = np.array(prof_arr)
        profhit_arr = np.array(profhit_arr)
        profsub_arr = np.array(profsub_arr)
        profsubhit_arr = np.array(profsubhit_arr)

        prof = (np.sum(prof_arr * profhit_arr, axis=0) / np.sum(profhit_arr, axis=0))
        profsub = (np.sum(profsub_arr * profsubhit_arr, axis=0) / np.sum(profsubhit_arr, axis=0))  

        profjack_arr = np.zeros_like(prof_arr)
        profsubjack_arr = np.zeros_like(profsub_arr)

        for isub in range(Nsub):
            proftot = np.sum(prof_arr * profhit_arr, axis=0)
            profi = prof_arr[isub] * profhit_arr[isub]
            hittot = np.sum(profhit_arr, axis=0)
            hiti = profhit_arr[isub]
            profjack_arr[isub] = (proftot - profi) / (hittot - hiti)

            proftot = np.sum(profsub_arr * profsubhit_arr, axis=0)
            profi = profsub_arr[isub] * profsubhit_arr[isub]
            hittot = np.sum(profsubhit_arr, axis=0)
            hiti = profsubhit_arr[isub]    
            profsubjack_arr[isub] = (proftot - profi) / (hittot - hiti)

        cov = np.zeros([len(prof),len(prof)])
        for i in range(len(prof)):
            for j in range(len(prof)):
                cov[i,j] = np.mean(profjack_arr[:,i]*profjack_arr[:,j]) \
                - np.mean(profjack_arr[:,i])*np.mean(profjack_arr[:,j])
        cov *= (Nsub-1)

        covsub = np.zeros([len(profsub),len(profsub)])
        for i in range(len(profsub)):
            for j in range(len(profsub)):
                covsub[i,j] = np.mean(profsubjack_arr[:,i]*profsubjack_arr[:,j]) \
                - np.mean(profsubjack_arr[:,i])*np.mean(profsubjack_arr[:,j])
        covsub *= (Nsub-1)

        psfdata[ifield] = {}
        psfdata[ifield]['Nsrc'] = len(xs)
        psfdata[ifield]['rbins'] = stackdat['rbins'].copy()
        psfdata[ifield]['rbinedges'] = stackdat['rbinedges'].copy()
        psfdata[ifield]['rsubbins'] = stackdat['rsubbins'].copy()
        psfdata[ifield]['rsubbinedges'] = stackdat['rsubbinedges'].copy()
        psfdata[ifield]['prof'] = prof
        psfdata[ifield]['profsub'] = profsub
        psfdata[ifield]['profhit'] = np.sum(profhit_arr,axis=0)
        psfdata[ifield]['prof_err'] = np.sqrt(np.diag(cov))
        psfdata[ifield]['profsub_err'] = np.sqrt(np.diag(covsub))
        psfdata[ifield]['cov'] = cov
        psfdata[ifield]['covsub'] = covsub
        if save_stackmap:
            psfdata[ifield]['stackmap'] = stack

    if savedata:
        fname = mypaths['alldat'] + 'TM'+ str(inst) + '/psfdata.pkl'
        with open(fname, "wb") as f:
            pickle.dump(psfdata, f)
            
        return

    return psfdata
from srcmap import *
from stack_ancillary import *
import bz2

def get_micecat_df(icat, add_Rvir=False):
    ira, idec = icat%45, icat//45 
    ra_min, ra_max = ira*2, ira*2+2
    dec_min, dec_max = idec*2, idec*2+2
    
    print('MICECAT field %d, %d < ra < %d, %d < dec < %d'%(icat,
                                                           ra_min, ra_max, 
                                                           dec_min, dec_max))
    
    fname = 'ra%d_%d_dec%d_%d'%(ra_min//10*10, ra_min//10*10+10, 
                                dec_min//4*4, dec_min//4*4+4)
    
    fname = mypaths['MCcatdat'] + fname + '.csv.bz2'
    with bz2.BZ2File(fname) as catalog_fd:
        df = pd.read_csv(catalog_fd, sep=",", index_col=False,
                         comment='#', na_values=r'\N')

    index = df[(df['ra_gal'] <= ra_min) | (df['ra_gal'] > ra_max) |\
              (df['dec_gal'] <= dec_min) | (df['dec_gal'] > dec_max)].index
    df.drop(index , inplace=True)

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

    df['x'] = (df['ra_gal'] - ra_min)*3600 / 7 - 1.5
    df['y'] = (df['dec_gal'] - dec_min)*3600 / 7 - 1.5

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

def run_micecat_fliter_test(inst, icat, filt_order_arr=[0,1,2,3,4,7,10,13],
                           savedir = './micecat_data/', save_data = True):
    
    mag_th = 20
    df = get_micecat_df(icat)
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
    
    data = np.zeros([len(filt_order_arr), 4, 25])
    datasub = np.zeros([len(filt_order_arr), 4, 15])
    for ifilt, filt_order in enumerate(filt_order_arr):
        print('cat #%d, %d-th order filter'%(icat,filt_order))
        if filt_order==0:
            filtmap = srcmap - np.mean(srcmap[mask==1])
        else:   
            filtmap = image_poly_filter(srcmap, mask, degree=filt_order)

        for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):
            #print('cat #%d, %d-th order filter, stack %d < m < %d'\
            #      %(icat,filt_order, m_min, m_max))
            spm = np.where((ms>=m_min) & (ms<m_max) &\
                           (xs>0) & (xs<1023) &\
                           (ys>0) & (ys<1023))[0]

            stack_class = stacking_mock(inst)
            stack_class.xls = xs[spm]
            stack_class.yls = ys[spm]
            stack_class.ms = ms[spm]

            stackdat = stack_class.run_stacking_bigpix(filtmap, mask, num, verbose=False)
            #stackdat = stack_class.run_stacking(filtmap, mask, num, verbose=False)

            data[ifilt, im, :] = stackdat['prof']
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
        fname  = savedir + 'filter_test_TM%d_icat%d.pkl'%(inst, icat)
        
        with open(fname, "wb") as f:
            pickle.dump(data_dict , f)
    
        # with open(fname,"rb") as f:
        #    data_dict = pickle.load(f)
    
    return data_dict

def run_micecat_fliter_test_cen(inst, icat, filt_order_arr=[0,1,2,3,4,5,6,7,10,13],
                           savedir = './micecat_data/', save_data = True):
    
    df = get_micecat_df(icat)
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
        filtmap = image_poly_filter(srcmap, mask, degree=filt_order)

        for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):
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

def run_micecat_1h(inst, icat, Nstack=500, Mhcut=np.inf, R200cut=np.inf, zcut=0,
                   savedir='./micecat_data/', save_data=True):
    '''
    Nstack: only stack 2 at most Nstack sources to speed up
    Mhcut [M_sun]: don't stack on source with Mh > Mhcut 
    R200cut [arcsec]: don't stack on the source with R200 > R200cut
    zcut: only stack sources with z > zcut
    '''
    
    df = get_micecat_df(icat)
    df = df.sort_values(by=['unique_halo_id'])
    z_arr = np.array(df['z_cgal'])
    Mh_arr = 10**np.array(df['lmhalo'])
    rhoc_arr = np.array(cosmo.critical_density(z_arr).to(u.M_sun / u.Mpc**3))
    rvir_arr = ((3 * Mh_arr) / (4 * np.pi * 200 * rhoc_arr))**(1./3)
    DA_arr = np.array(cosmo.comoving_distance(z_arr))
    rvir_ang_arr = (rvir_arr / DA_arr) * u.rad.to(u.arcsec)
    df['Rv_arcsec'] = rvir_ang_arr

    make_srcmap_class = make_srcmap(inst)
    stack_class = stacking_mock(inst)

    df1h = {}
    for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):
        df1h[im] = {}
        dfm = df[(df['I']>=m_min) & (df['I']<m_max) \
                 & (df['x']<1023.5) & (df['x']>-0.5)\
                & (df['y']<1023.5) & (df['y']>-0.5) \
                & (df['z_cgal']>zcut)].copy()
        galids = np.array(dfm.index)
        haloids = dfm['unique_halo_id'].values
        shuffle_idx = np.random.permutation(len(dfm))
        galids, haloids = galids[shuffle_idx], haloids[shuffle_idx]
        dfm1h = pd.DataFrame()
        galid_removed_list = []
        for i, (haloid, galid) in enumerate(zip(haloids, galids)):
            dfi = df[df['unique_halo_id']==haloid].copy()
            dfi['stack_gal_id'] = galid
            Rvi = np.mean(dfi['Rv_arcsec'].values)
            Mh = 10**np.mean(dfi['lmhalo'].values)
            if (Rvi > R200cut) and (Mh > Mhcut):
                galid_removed_list.append(galid)
                continue
            dfi.drop(['unique_halo_id', 'z_cgal', 'nsats', 'lmhalo'], axis=1, inplace=True)
            dfi.drop(galid, inplace=True)
            dfm1h = pd.concat([dfm1h, dfi])
        dfm.drop(galid_removed_list, inplace=True)
        df1h[im] = {'dfm': dfm, 'df1h':dfm1h}

    # data = np.zeros([4, 25])
    # datasub = np.zeros([4, 15])
    # for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):
    #     dfm, dfm1h = df1h[im]['dfm'], df1h[im]['df1h']

    #     start_time = time.time()
    #     mapstack, maskstack = 0., 0.
    #     for i, galid in enumerate(dfm.index):
    #         if i > Nstack:
    #             break
    #         dfi = dfm1h.loc[dfm1h['stack_gal_id']==galid]
    #         x0, y0, m0 = dfm.loc[galid][['x','y','I']]

    #         if i%100==0:
    #             print('stack 1-halo, icat %d, %d < m < %d, %d / %d, %d sats, t = %.2f min'\
    #                   %(icat, m_min, m_max, i, len(dfm), len(dfi), (time.time()-start_time)/60))

    #         make_srcmap_class.xls = np.array(dfi['x'])
    #         make_srcmap_class.yls = np.array(dfi['y'])
    #         make_srcmap_class.ms = np.array(dfi['I'])

    #         if inst == 1:
    #             make_srcmap_class.ms_inband = np.array(dfi['I'])
    #         else:
    #             make_srcmap_class.ms_inband = np.array(dfi['H'])

    #         srcmapi = make_srcmap_class.run_srcmap(ptsrc=True, verbose=False)

    #         maski, numi = Ith_mask_mock(np.concatenate((np.array(dfi['x']),np.array([x0]))),
    #                                     np.concatenate((np.array(dfi['y']),np.array([y0]))),
    #                                     np.concatenate((np.array(dfi['I']),np.array([m0]))),
    #                                     verbose=False)      

    #         stack_class.xls = np.array([x0])
    #         stack_class.yls = np.array([y0])
    #         stack_class.ms = np.array([m0])

    #         _, maskstacki, mapstacki = stack_class.run_stacking(srcmapi, maski, numi,
    #                                                            verbose=False, return_profile=False)

    #         mapstack += mapstacki
    #         maskstack += maskstacki

    #     stack = np.zeros_like(mapstack)
    #     sp = np.where(maskstack!=0)
    #     stack[sp] = mapstack[sp] / maskstack[sp]
    #     stack[maskstack==0] = 0

    #     dx = stack_class.dx
    #     profile = radial_prof(np.ones([2*dx+1,2*dx+1]), dx, dx)
    #     rbinedges, rbins = profile['rbinedges'], profile['rbins']
    #     rsubbins, rsubbinedges = radial_binning(rbins, rbinedges)
    #     Nbins = len(rbins)
    #     Nsubbins = len(rsubbins)
    #     radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx)
        
    #     prof_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
    #     for ibin in range(Nbins):
    #         spi = np.where((radmapstamp>=rbinedges[ibin]) &\
    #                        (radmapstamp<rbinedges[ibin+1]))
    #         prof_arr[ibin] += np.sum(mapstack[spi])
    #         hit_arr[ibin] += np.sum(maskstack[spi])
    #     prof_norm = np.zeros_like(prof_arr)
    #     prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]
    #     data[im,:] = prof_norm
    
    #     prof_arr, hit_arr = np.zeros(Nsubbins), np.zeros(Nsubbins)
    #     for ibin in range(Nsubbins):
    #         spi = np.where((radmapstamp>=rsubbinedges[ibin]) &\
    #                        (radmapstamp<rsubbinedges[ibin+1]))
    #         prof_arr[ibin] += np.sum(mapstack[spi])
    #         hit_arr[ibin] += np.sum(maskstack[spi])
    #     prof_norm = np.zeros_like(prof_arr)
    #     prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]
    #     datasub[im,:] = prof_norm

    # rbins = rbins*0.7
    # rbinedges = rbinedges*0.7 
    # rsubbins = rsubbins*0.7
    # rsubbinedges = rsubbinedges*0.7 

    # data_dict = {'data': data, 'datasub': datasub, 
    #             'rbins':rbins, 'rbinedges':rbinedges, 
    #             'rsubbins':rsubbins, 'rsubbinedges':rsubbinedges}
    data_dict = {}####
    if save_data:
        fname  = savedir + 'onehalo_TM%d_icat%d.pkl'%(inst, icat)
        if (Mhcut != np.inf) or (R200cut != np.inf):
            fname  = savedir + 'onehalo_TM{:.0f}_icat{:.0f}_R200cut{:.0f}_Mhcut{:.0f}.pkl'\
            .format(inst, icat, R200cut, np.log10(Mhcut))
        if zcut!=0:
            fname  = savedir + 'onehalo_TM{:.0f}_icat{:.0f}_zcut{:.2f}.pkl'\
            .format(inst, icat, zcut)
            if (Mhcut != np.inf) or (R200cut != np.inf):
                fname  = savedir + 'onehalo_TM{:.0f}_icat{:.0f}_zcut{:.2f}_R200cut{:.0f}_Mhcut{:.0f}.pkl'\
                .format(inst, icat, zcut, R200cut, np.log10(Mhcut))
        with open(fname, "wb") as f:
            pickle.dump(data_dict , f)
    
    #return data_dict####
    print(fname)####
    return df1h###

def run_micecat_batch(inst, ibatch, run_type='all', return_data=False, **kwargs):
    
    if return_data:
        data_dicts = [] 
    icat_arr = np.linspace(0,9,10) + ibatch*10
    for icat in icat_arr:

        if run_type == 'all':
            data_dict = run_micecat_fliter_test(inst, icat, **kwargs)
        elif run_type == 'cen':
            data_dict = run_micecat_fliter_test_cen(inst, icat, **kwargs)
        elif run_type == '1h':
            data_dict = run_micecat_1h(inst, icat, **kwargs)

        if return_data:
            data_dicts.append(data_dict)

    if return_data:
        return data_dicts

    return

def get_micecat_sim_1h(inst, im, Mhcut=np.inf, R200cut=np.inf, zcut=0, sub=False):
    '''
    Get the MICECAT 1halo sim results.
    '''
    savedir='./micecat_data/'
    data_all = []
    for icat in range(90):
        fname  = 'onehalo_TM%d_icat%d.pkl'%(inst, icat)
        if (Mhcut != np.inf) or (R200cut != np.inf):
            fname  = 'onehalo_TM{:.0f}_icat{:.0f}_R200cut{:.0f}_Mhcut{:.0f}.pkl'\
            .format(inst, icat, R200cut, np.log10(Mhcut))
        if zcut!=0:
            fname  = 'onehalo_TM{:.0f}_icat{:.0f}_zcut{:.2f}.pkl'\
            .format(inst, icat, zcut)

        if fname not in os.listdir(savedir):
            continue

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
    
    data_all = np.array(data_all)[:,im,:]
    
    data_avg = np.mean(data_all, axis=0)
    data_std = np.std(data_all, axis=0)
    
    return rbins, data_avg, data_std, data_all

def get_micecat_sim_cen(inst, im, sub=False, 
    filt_order=None, ratio=False, return_icat=False):
    '''
    Get the MICECAT central gal sim results.
    '''
    savedir='./micecat_data/'
    typename = 'filter_test_cen'
    data_all = []
    icat_arr = []
    for icat in range(90):
        savedir='./micecat_data/'
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


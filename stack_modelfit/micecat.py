from srcmap import *
from stack_ancillary import *
import bz2

def get_micecat_df(icat):
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

    df['I'] = df['euclid_nisp_y_true']
    Hmag = 0.5*(10**(-df['euclid_nisp_j_true']/2.5) + 10**(-df['euclid_nisp_h_true']/2.5))
    Hmag = -2.5*np.log10(Hmag)
    df['H'] = Hmag

    df['x'] = (df['ra_gal'] - ra_min)*3600 / 7 - 1.5
    df['y'] = (df['dec_gal'] - dec_min)*3600 / 7 - 1.5

    df.drop(['lsfr','lmstellar', 'ra_gal', 'dec_gal','euclid_nisp_y_true',
             'euclid_nisp_j_true','euclid_nisp_h_true'], axis=1, inplace=True)
    
    return df

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
    for ifilt, filt_order in enumerate(filt_order_arr):
        print('cat #%d, %d-th order filter'%(icat,filt_order))
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
    
    rbins = stackdat['rbins']
    if save_data:
        data_dict = {'data': data, 'rbins':rbins,
                    'filt_order_arr':filt_order_arr}
        fname  = savedir + 'filter_test_TM%d_icat%d.pkl'%(inst, icat)
        
        with open(fname, "wb") as f:
            pickle.dump(data_dict , f)
    
        # with open(fname,"rb") as f:
        #    data_dict = pickle.load(f)
    
    return data, rbins, filt_order_arr
from micecat import *
from srcmap import *
from mask import *

def run_f1h_test(icat=0, Nsamp = 100):
    make_srcmap_class1 = make_srcmap(1)
    make_srcmap_class2 = make_srcmap(2)
    df = get_micecat_df(icat)

    df['Fnu_I'] = 3631 * 10**(-df['I'] / 2.5)
    df['Fnu_H'] = 3631 * 10**(-df['H'] / 2.5)
    nuFnu_I = np.array(make_srcmap_class1._ABmag2Iciber(df['I'].copy()))
    nuFnu_H = np.array(make_srcmap_class2._ABmag2Iciber(df['H'].copy()))

    df_dict = {}
    for icase,(im,M_min,M_max,name) in enumerate(zip([1,2,3,2,3],
                                            [-23,-23,-23,-22,-22],
                                            [-22,-22,-22,-21,-21],
        ['high-M/low-z','high-M/med-z','high-M/high-z','low-M/low-z','low-M/high-z'])):
        m_min, m_max = magbindict['m_min'][im], magbindict['m_max'][im]
        df_dict[icase] = {'m_min':m_min, 'm_max':m_max, 'M_min':M_min, 'M_max':M_max, 'name':name}

        idx_c = np.where((df['I'] < m_max) & (df['I'] > m_min) &(df['M_I'] < M_max) &\
                         (df['z_cgal'] > 0.15) &\
                         (df['M_I'] > M_min) & (df['flag_central']==0))[0]
        idx_s = np.where((df['I'] < m_max) & (df['I'] > m_min) &(df['M_I'] < M_max) &\
                         (df['z_cgal'] > 0.15) &\
                         (df['M_I'] > M_min) & (df['flag_central']==1))[0]

        df1h = pd.DataFrame()
        for i, (haloid, galid) in enumerate(zip(df['unique_halo_id'].iloc[idx_c].values,
                                                df.index[idx_c].values)):
            dfi = df[df['unique_halo_id']==haloid].copy()
            dfi['stack_gal_id'] = galid
            df1h = pd.concat([df1h, dfi])
        df_dict[icase]['df_c'], df_dict[icase]['df1h_c'] = df.iloc[idx_c], df1h

        df1h = pd.DataFrame()
        for i, (haloid, galid) in enumerate(zip(df['unique_halo_id'].iloc[idx_s].values,
                                                df.index[idx_s].values)):
            dfi = df[df['unique_halo_id']==haloid].copy()
            dfi['stack_gal_id'] = galid
            df1h = pd.concat([df1h, dfi])
        df_dict[icase]['df_s'], df_dict[icase]['df1h_s'] = df.iloc[idx_s], df1h
        
        print('case #%d, %d cen, %d sats'%(icase, len(df_dict[icase]['df_c']), 
                                           len(df_dict[icase]['df_s'])))
    f1h_dict = {}
    for icase in range(5):
        f1h_dict[icase] = {}
        for typename in ['c','s']:
            dfm, dfm1h = df_dict[icase]['df_'+typename], df_dict[icase]['df1h_'+typename]
            f1h_dict[icase]['f1h_I_'+typename] = np.zeros(len(dfm))
            f1h_dict[icase]['f1h_H_'+typename] = np.zeros(len(dfm))
            f1h_dict[icase]['f1h_I_masked_'+typename] = np.zeros(len(dfm))
            f1h_dict[icase]['f1h_H_masked_'+typename] = np.zeros(len(dfm))
            start_time = time.time()
            for i, galid in enumerate(dfm.index[np.random.permutation(len(dfm))]):
                if i > Nsamp:
                    break

                if i%(np.min((len(dfm),Nsamp))//10)==0:
                    print('Calculate f_1h for #%d case,  %d / %d (%.1f %%),t = %.2f min'\
                          %(icase, i, np.min((len(dfm),Nsamp)), i/np.min((len(dfm),Nsamp))*100,
                            (time.time()-start_time)/60))

                Fsrc_I = dfm['Fnu_I'].loc[dfm.index==galid].values[0]
                Fsrc_H = dfm['Fnu_H'].loc[dfm.index==galid].values[0]
                dfi = dfm1h.loc[dfm1h['stack_gal_id']==galid]
                Ftot_I = np.sum(dfi['Fnu_I'])
                Ftot_H = np.sum(dfi['Fnu_H'])

                f_res1,f_res2 = np.ones(len(dfi)),np.ones(len(dfi))
                for spmk in np.where(dfi['I'] < 20)[0]:
                    if dfi.index[spmk] == galid:
                        continue
                    x,y,I = dfi[['x','y','I']].iloc[spmk]
                    make_srcmap_class1.xls = np.array([x])
                    make_srcmap_class1.yls = np.array([y])
                    make_srcmap_class1.ms = np.array([I])
                    make_srcmap_class2.xls = np.array([x])
                    make_srcmap_class2.yls = np.array([y])
                    make_srcmap_class2.ms = np.array([I])

                    maski, _ = Ith_mask_mock(np.array([x]), np.array([y]), 
                                             np.array([I]), verbose=False)    

                    make_srcmap_class1.ms_inband = np.array(dfi['I'])
                    srcmapi_I = make_srcmap_class1.run_srcmap(ptsrc=True, verbose=False)
                    make_srcmap_class2.ms_inband = np.array(dfi['H'])
                    srcmapi_H = make_srcmap_class2.run_srcmap(ptsrc=True, verbose=False)

                    f_res1[spmk] = np.sum(srcmapi_I*maski) / np.sum(srcmapi_I)
                    f_res2[spmk] = np.sum(srcmapi_H*maski) / np.sum(srcmapi_H)

                Ftot_I_masked = np.sum(dfi['Fnu_I']*f_res1)
                Ftot_H_masked = np.sum(dfi['Fnu_H']*f_res2)

                f1h_dict[icase]['f1h_I_'+typename][i] = 1 - Fsrc_I/Ftot_I
                f1h_dict[icase]['f1h_H_'+typename][i] = 1 - Fsrc_H/Ftot_H
                f1h_dict[icase]['f1h_I_masked_'+typename][i] = 1 - Fsrc_I/Ftot_I_masked
                f1h_dict[icase]['f1h_H_masked_'+typename][i] = 1 - Fsrc_I/Ftot_H_masked
            f1h_dict[icase]['f1h_I_'+typename] = f1h_dict[icase]['f1h_I_'+typename][:Nsamp]
            f1h_dict[icase]['f1h_H_'+typename] = f1h_dict[icase]['f1h_H_'+typename][:Nsamp]
            f1h_dict[icase]['f1h_I_masked_'+typename] = f1h_dict[icase]['f1h_I_masked_'+typename][:Nsamp]
            f1h_dict[icase]['f1h_H_masked_'+typename] = f1h_dict[icase]['f1h_H_masked_'+typename][:Nsamp]
            
    fname = 'micecat_data/f1h_test_icat%d.pkl'%(icat)
    with open(fname, "wb") as f:
        pickle.dump(f1h_dict , f)
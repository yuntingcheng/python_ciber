from srcmap import *
from mask import *
from power_spec import *
from micecat import *

def run_1h_power_spec_test(ibatch, m_max_vega=17, mag_th_psf = 20):
    batch_size = 10
    icat_arr = np.linspace(0, batch_size-1, batch_size) + ibatch*batch_size
    icat_arr = icat_arr.astype(int)

    mag_th = m_max_vega - 2.5*np.log10(1594./3631.)
    make_srcmap_class1 = make_srcmap(1)
    make_srcmap_class2 = make_srcmap(2)

    data = {'icat_arr':icat_arr}
    for icat in icat_arr:
        print('icat %d'%icat)
        data[icat] = {}
        df = get_micecat_df(icat)
        df['Fnu_I'] = 3631 * 10**(-df['I'] / 2.5)
        df['Fnu_H'] = 3631 * 10**(-df['H'] / 2.5)
        nuFnu_I = np.array(make_srcmap_class1._ABmag2Iciber(df['I'].copy()))
        nuFnu_H = np.array(make_srcmap_class2._ABmag2Iciber(df['H'].copy()))
        idx_b = np.where(df['I'] < mag_th)[0]
        f_res1,f_res2 = np.ones(len(df)),np.ones(len(df))
        
        start_time = time.time()
        for i,idx in enumerate(idx_b):
            if i%(len(idx_b)//10)==0:
                print('Calculate residual flux of masked sources %d / %d (%.1f %%),t = %.2f min'\
                      %(i, len(idx_b), i/len(idx_b)*100, (time.time()-start_time)/60))
            dfi = df.iloc[idx]
            mask,_ = MZ14_mask(1,np.array([dfi['x']]),np.array([dfi['y']]),
                                 np.array([dfi['I']]),m_max_vega=m_max_vega,verbose=True)
            make_srcmap_class1.ms = np.array([dfi['I']])
            make_srcmap_class1.xls = np.array([dfi['x']])
            make_srcmap_class1.yls = np.array([dfi['y']])
            make_srcmap_class2.ms = np.array([dfi['I']])
            make_srcmap_class2.xls = np.array([dfi['x']])
            make_srcmap_class2.yls = np.array([dfi['y']])
            make_srcmap_class1.ms_inband = np.array([dfi['I']])
            make_srcmap_class2.ms_inband = np.array([dfi['H']])

            srcmapb1 = make_srcmap_class1.run_srcmap(ptsrc=True, verbose=False)
            srcmapb2 = make_srcmap_class2.run_srcmap(ptsrc=True, verbose=False)
            f_res1[idx] = np.sum(srcmapb1[mask==1]) / nuFnu_I[idx]
            f_res2[idx] = np.sum(srcmapb2[mask==1]) / nuFnu_H[idx]

        df['Fnu_I_res'] = df['Fnu_I'].copy() * f_res1
        df['Fnu_H_res'] = df['Fnu_H'].copy() * f_res2        
        dfc = df.loc[df['flag_central']==0]
        dfsum = df.groupby('unique_halo_id')[['Fnu_I_res','Fnu_H_res']].sum()
        dfsum.rename(columns={'Fnu_I_res':'Fnu_I_res_sum','Fnu_H_res':'Fnu_H_res_sum'}, inplace=True)
        dfc = dfc.join(dfsum, on='unique_halo_id', how='inner')
        dfc['I'] = -2.5 * np.log10(dfc['Fnu_I_res_sum']/3631)
        dfc['H'] = -2.5 * np.log10(dfc['Fnu_H_res_sum']/3631) 
        
        for i,(name,dfi) in enumerate(zip(['full','cen'],[df,dfc])):
            data[icat][name] = {}
            xs, ys, ms = np.array(dfi['x']), np.array(dfi['y']), np.array(dfi['I'])
            
            if name == 'full':
                print('making mask for %s'%name)
                mask,num = MZ14_mask(1,xs,ys,ms,m_max_vega=m_max_vega,verbose=False)

                print('get mkk for %s'%name)
                mask_mkk = mask_Mkk(mask)
                mask_mkk.get_Mkk_sim(Nsims=10,verbose=False)

            spb = np.where(dfi['I']<=mag_th_psf)[0]
            spf = np.where(dfi['I']>mag_th_psf)[0]
            for inst in [1,2]:
                data[icat][name][inst] = {}
                ms_inband = np.array(dfi['I']) if inst==1 else np.array(dfi['H'])

                print('making srcmap for %s TM %d'%(name,inst))
                make_srcmap_class = make_srcmap(inst)

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

                if name == 'cen':
                    l,Cl,Clerr = get_power_spec(srcmap)
                else:
                    l,Cl0,Cl0err = get_power_spec(srcmap, mask=mask)
                    Cl, Clerr = mask_mkk.Mkk_correction(Cl0, Clerr=Cl0err)
                
                data[icat][name][inst]['Cl'] = Cl
                data[icat][name][inst]['Clerr'] = Clerr
                    
        data['l'] = l 
        fname = 'micecat_data/PS1h_test_ibatch%d_maskth%d.pkl'%(ibatch,m_max_vega)
        with open(fname, "wb") as f:
            pickle.dump(data , f)


# def run_1h_power_spec_test(ibatch):
#     batch_size = 10
#     icat_arr = np.linspace(0, batch_size-1, batch_size) + ibatch*batch_size
#     icat_arr = icat_arr.astype(int)
    
#     data = {'icat_arr':icat_arr}
#     for icat in icat_arr:
#         print('icat %d'%icat)
#         data[icat] = {}
#         df = get_micecat_df(icat)
#         dfc = df.loc[df['flag_central']==0]
#         dfs = df.copy()
#         dfs['Fnu_I'] = 3631 * 10**(-df['I'] / 2.5)
#         dfs['Fnu_H'] = 3631 * 10**(-df['H'] / 2.5)
#         dfs1 = dfs.groupby('unique_halo_id')[['Fnu_I','Fnu_H']].sum()
#         dfc = dfc.join(dfs1, on='unique_halo_id', how='inner')
#         dfc['I'] = -2.5 * np.log10(dfc['Fnu_I']/3631)
#         dfc['H'] = -2.5 * np.log10(dfc['Fnu_H']/3631)

#         mag_th = 20
#         for i,(name,dfi) in enumerate(zip(['full','cen'],[df,dfc])):
#             data[icat][name] = {}
#             xs, ys, ms = np.array(dfi['x']), np.array(dfi['y']), np.array(dfi['I'])

#             print('making mask for %s'%name)
#             mask,num = MZ14_mask(1,xs,ys,ms,verbose=False)

#             print('get mkk for %s'%name)
#             mask_mkk = mask_Mkk(mask)
#             mask_mkk.get_Mkk_sim(Nsims=50,verbose=False)
#             spb = np.where(dfi['I']<=mag_th)[0]
#             spf = np.where(dfi['I']>mag_th)[0]

#             for inst in [1,2]:
#                 data[icat][name][inst] = {}
#                 ms_inband = np.array(dfi['I']) if inst==1 else np.array(dfi['H'])

#                 print('making srcmap for %s TM %d'%(name,inst))
#                 make_srcmap_class = make_srcmap(inst)

#                 make_srcmap_class.ms = ms[spb]
#                 make_srcmap_class.ms_inband = ms_inband[spb]
#                 make_srcmap_class.xls = xs[spb]
#                 make_srcmap_class.yls = ys[spb]
#                 srcmapb = make_srcmap_class.run_srcmap(ptsrc=True, verbose=False)

#                 make_srcmap_class.ms = ms[spf]
#                 make_srcmap_class.ms_inband = ms[spf]
#                 make_srcmap_class.xls = xs[spf]
#                 make_srcmap_class.yls = ys[spf]
#                 srcmapf = make_srcmap_class.run_srcmap_nopsf()
#                 srcmap = srcmapb + srcmapf

#                 l,Cl0,Cl0err = get_power_spec(srcmap, mask=mask)
#                 Cl, Clerr = mask_mkk.Mkk_correction(Cl0, Clerr=Cl0err)
#                 Dl0 = np.sqrt(Cl0*l*(l+1)/2/np.pi)
#                 Dl0err = np.sqrt(Cl0err*l*(l+1)/2/np.pi)
#                 Dl = np.sqrt(Cl*l*(l+1)/2/np.pi)
#                 Dlerr = np.sqrt(Clerr*l*(l+1)/2/np.pi)

#                 data[icat][name][inst]['l'] = l
#                 data[icat][name][inst]['Dl0'] = Dl0
#                 data[icat][name][inst]['Dl0err'] = Dl0err
#                 data[icat][name][inst]['Dl'] = Dl
#                 data[icat][name][inst]['Dlerr'] = Dlerr
    
#         fname = 'micecat_data/PS1h_test_ibatch%d.pkl'%(ibatch)
#         with open(fname, "wb") as f:
#             pickle.dump(data , f)
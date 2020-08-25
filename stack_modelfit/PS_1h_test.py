from srcmap import *
from mask import *
from power_spec import *
from micecat import *

def run_1h_power_spec_test(ibatch):
    batch_size = 10
    icat_arr = np.linspace(0, batch_size-1, batch_size) + ibatch*batch_size
    icat_arr = icat_arr.astype(int)
    
    data = {'icat_arr':icat_arr}
    for icat in icat_arr:
        print('icat %d'%icat)
        data[icat] = {}
        df = get_micecat_df(0)
        dfc = df.loc[df['flag_central']==0]
        dfs = df.copy()
        dfs['Fnu_I'] = 3631 * 10**(-df['I'] / 2.5)
        dfs['Fnu_H'] = 3631 * 10**(-df['H'] / 2.5)
        dfs1 = dfs.groupby('unique_halo_id')[['Fnu_I','Fnu_H']].sum()
        dfc = dfc.join(dfs1, on='unique_halo_id', how='inner')
        dfc['I'] = -2.5 * np.log10(dfc['Fnu_I']/3631)
        dfc['H'] = -2.5 * np.log10(dfc['Fnu_H']/3631)

        mag_th = 20
        for i,(name,dfi) in enumerate(zip(['full','cen'],[df,dfc])):
            data[icat][name] = {}
            xs, ys, ms = np.array(dfi['x']), np.array(dfi['y']), np.array(dfi['I'])

            print('making mask for %s'%name)
            mask,num = MZ14_mask(1,xs,ys,ms,verbose=False)

            print('get mkk for %s'%name)
            mask_mkk = mask_Mkk(mask)
            mask_mkk.get_Mkk_sim(Nsims=50,verbose=False)
            spb = np.where(dfi['I']<=mag_th)[0]
            spf = np.where(dfi['I']>mag_th)[0]

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

                l,Cl0,Cl0err = get_power_spec(srcmap, mask=mask)
                Cl, Clerr = mask_mkk.Mkk_correction(Cl0, Clerr=Cl0err)
                Dl0 = np.sqrt(Cl0*l*(l+1)/2/np.pi)
                Dl0err = np.sqrt(Cl0err*l*(l+1)/2/np.pi)
                Dl = np.sqrt(Cl*l*(l+1)/2/np.pi)
                Dlerr = np.sqrt(Clerr*l*(l+1)/2/np.pi)

                data[icat][name][inst]['l'] = l
                data[icat][name][inst]['Dl0'] = Dl0
                data[icat][name][inst]['Dl0err'] = Dl0err
                data[icat][name][inst]['Dl'] = Dl
                data[icat][name][inst]['Dlerr'] = Dlerr
    
        fname = 'micecat_data/PS1h_test_ibatch%d.pkl'%(ibatch)
        with open(fname, "wb") as f:
        pickle.dump(data , f)
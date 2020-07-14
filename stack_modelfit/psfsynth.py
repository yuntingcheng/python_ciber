from reduction import *
from stack import * 
from psfstack import *
from scipy import interpolate

def run_psf_synth(inst, ifield, filt_order=3, savedata=True):

    # fname = mypaths['alldat'] + 'TM'+ str(inst) + \
    # '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    # with open(fname, "rb") as f:
    #     profdat = pickle.load(f)

    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    cal = -cal_factor_dict['apf2nWpm2psr'][inst][ifield]
    
    psfdata_out = stack_psf(inst, data_maps[inst].stackmapdat,m_min=4, m_max=9,
     ifield_arr=[ifield], Nsub_single=True, savedata=False, save_stackmap=False)

    profdat = {}
    profdat['rbins'] = psfdata_out[ifield]['rbins']
    profdat['rbinedges'] = psfdata_out[ifield]['rbinedges']
    profdat['rsubbins'] = psfdata_out[ifield]['rsubbins']
    profdat['rsubbinedges'] = psfdata_out[ifield]['rsubbinedges']
    profdat['filt_order'] = filt_order

    profdat['out'] = {}
    profdat['out']['m_min'] = 4
    profdat['out']['m_max'] = 9
    profdat['out']['Nsrc'] = psfdata_out[ifield]['Nsrc']
    profdat['out']['profcb'] = psfdata_out[ifield]['prof']*cal
    profdat['out']['profcb_err'] = psfdata_out[ifield]['prof_err']*cal
    profdat['out']['profcbsub'] = psfdata_out[ifield]['profsub']*cal
    profdat['out']['profcbsub_err'] = psfdata_out[ifield]['profsub_err']*cal
    profdat['out']['cov'] = psfdata_out[ifield]['cov']*cal**2
    profdat['out']['covsub'] = psfdata_out[ifield]['covsub']*cal**2

    psfdata_mid = stack_psf(inst, data_maps[inst].stackmapdat, m_min=13, m_max=14,
     ifield_arr=[ifield], Nsub_single=True, savedata=False, save_stackmap=False)

    profdat['mid'] = {}
    profdat['mid']['m_min'] = 13
    profdat['mid']['m_max'] = 14
    profdat['mid']['Nsrc'] = psfdata_mid[ifield]['Nsrc']
    profdat['mid']['profcb'] = psfdata_mid[ifield]['prof']*cal
    profdat['mid']['profcb_err'] = psfdata_mid[ifield]['prof_err']*cal
    profdat['mid']['profcbsub'] = psfdata_mid[ifield]['profsub']*cal
    profdat['mid']['profcbsub_err'] = psfdata_mid[ifield]['profsub_err']*cal
    profdat['mid']['cov'] = psfdata_mid[ifield]['cov']*cal**2
    profdat['mid']['covsub'] = psfdata_mid[ifield]['covsub']*cal**2

    if savedata:
        fname = mypaths['alldat'] + 'TM'+ str(inst) +\
         '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)
    
    mapin, strmask, strnum, mask_inst1, mask_inst2 = \
    load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                       (inst,ifield,'strmask'), 
                                       (inst,ifield,'strnum'),
                                       (1,ifield,'mask_inst'),
                                       (2,ifield,'mask_inst')])
    
    for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):

        stack_class = stacking(inst, ifield, m_min, m_max, filt_order=filt_order, 
                            load_from_file=True,BGsub=False)

        cliplim = stack_class._stackihl_PS_cliplim()

        srcdat = ps_src_select(inst, ifield, m_min, m_max, 
            [mask_inst1, mask_inst2], sample_type='jack_region')

        stackdat = stack_class.stack_PS(srctype='s',cliplim=cliplim, 
                                        srcdat=srcdat, verbose=False)
        stack_class.stackdat = stackdat
        stack_class._get_jackknife_profile()
        stack_class._get_covariance()

        profdat[im] = {}
        profdat[im]['m_min'] = m_min
        profdat[im]['m_max'] = m_max
        profdat[im]['Nsrc'] = stackdat['Nsrc']
        profdat[im]['profcb'] = stack_class.stackdat['profcb']
        profdat[im]['profcb_err'] = np.sqrt(np.diag(stackdat['cov']['profcb']))
        profdat[im]['profcbsub'] = stack_class.stackdat['profcbsub']
        profdat[im]['profcbsub_err'] = np.sqrt(np.diag(stackdat['cov']['profcbsub']))
        profdat[im]['cov'] = stackdat['cov']['profcb']
        profdat[im]['covsub'] = stackdat['cov']['profcbsub']
        profdat[im]['sub'] = stackdat['sub']

        if savedata:
            fname = mypaths['alldat'] + 'TM'+ str(inst) +\
             '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
            with open(fname, "wb") as f:
                pickle.dump(profdat, f)
        
    if savedata:
        return
    
    return profdat

def run_psf_combine(inst, ifield, savedata=True):

    profc = np.zeros(25)
    profcsub = np.zeros(15)

    m_min, m_max = 13, 14
    if inst ==1 and ifield ==4:
        m_min, m_max = 14,15
    if inst == 2:
        m_min, m_max = 14,15
    fname = mypaths['alldat'] + 'TM'+ str(inst) +\
     '/psfdata_synth_ps_%s_%d_%d.pkl'%(fieldnamedict[ifield],m_min, m_max)

    with open(fname, "rb") as f:
        profdat = pickle.load(f)

    profc[:12] = profdat['profcb'][:12] / profdat['profcb'][0]
    profcsub[:7] = profdat['profcbsub'][:7] / profdat['profcb'][0]
    
    covc_stack = np.zeros([25,25])
    covcsub_stack = np.zeros([15,15])
    covc_stack[:12,:12] = profdat['cov'][:12,:12] / profdat['profcb'][0]**2
    covcsub_stack[:7,:7] = profdat['covsub'][:7,:7] / profdat['profcb'][0]**2
    
    m_min, m_max = 9, 10
    fname = mypaths['alldat'] + 'TM'+ str(inst) +\
     '/psfdata_synth_2m_%s_%d_%d.pkl'%(fieldnamedict[ifield],m_min, m_max)
    with open(fname, "rb") as f:
        profdat = pickle.load(f)
    slope_mid = np.polyfit(np.log10(profdat['rbins'][11:15]),
                       np.log10(profdat['profcb'][11:15]),1)[0]
    proffit = 10 ** (slope_mid * np.log10(profdat['rbins']))
    proffitsub = 10 ** (slope_mid * np.log10(profdat['rsubbins']))
    profc[11:14] = proffit[11:14] / proffit[11] * profc[11]
    profcsub[6:9] = proffitsub[6:9] / proffitsub[6] * profcsub[6]

    m_min, m_max = 4, 9
    fname = mypaths['alldat'] + 'TM'+ str(inst) +\
     '/psfdata_synth_2m_%s_%d_%d.pkl'%(fieldnamedict[ifield],m_min, m_max)
    with open(fname, "rb") as f:
        profdat = pickle.load(f)
    slope_out = np.polyfit(np.log10(profdat['rbins'][11:17]),
                       np.log10(profdat['profcb'][11:17]),1)[0]
    proffit = 10 ** (slope_out * np.log10(profdat['rbins']))
    proffitsub = 10 ** (slope_out * np.log10(profdat['rsubbins']))

    profc[13:] = proffit[13:] / proffit[13] * profc[13]
    profcsub[8:] = proffitsub[8:] / proffitsub[8] * profcsub[8]
    
    # propagate systematic offset on 1st and 11th bin to outer radii
    sqrtvar = np.sqrt(covc_stack[0,0]/profc[0]**2) * profc[:,np.newaxis]
    covc_scaling = sqrtvar@sqrtvar.T
    sqrtvar = np.sqrt(covc_stack[0,0]/profc[0]**2) * profcsub[:,np.newaxis]
    covcsub_scaling = sqrtvar@sqrtvar.T
    # ferr =  covc_stack[11,11]/profc[11]**2
    # covc_scaling[12:,12:] += ferr * (profc[12:,np.newaxis]@profc[12:,np.newaxis].T)
    # covcsub_scaling[6:,6:] += ferr * (profcsub[6:,np.newaxis]@profcsub[6:,np.newaxis].T)
    
    # systematic err from Gaia stack
    sys_err = np.zeros_like(profc)
    syssub_err = np.zeros_like(profcsub)
    for im,(m_min, m_max) in enumerate(zip(np.arange(17,20), np.arange(18,21))):
        fname = mypaths['alldat'] + 'TM'+ str(inst) +\
         '/psfdata_synth_gaia_%s_%d_%d.pkl'%(fieldnamedict[ifield],m_min, m_max)
        with open(fname, "rb") as f:
            profdat = pickle.load(f)
        sysi = np.abs((profdat['profcb']/profdat['profcb'][0]/profc) - 1)
        sys_err[(sysi>sys_err)] = sysi[(sysi>sys_err)]
        
        syssubi = np.abs((profdat['profcbsub']/profdat['profcb'][0]/profcsub) - 1)
        syssub_err[(syssubi>syssub_err)] = syssubi[(syssubi>syssub_err)]

    sys_err[9:] = sys_err[9]
    syssub_err[4:] = syssub_err[4]
    
    sqrtvar = sys_err * profc[:,np.newaxis]
    cov_gaia_sys = sqrtvar@sqrtvar.T
    sqrtvar = syssub_err * profcsub[:,np.newaxis]
    covsub_gaia_sys = sqrtvar@sqrtvar.T

    fname = mypaths['alldat'] + 'TM'+ str(inst) + \
    '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    with open(fname, "rb") as f:
        profdat = pickle.load(f)
    
    for im,(m_min,m_max) in enumerate(zip(magbindict['m_min'],magbindict['m_max'])):
        profdat[im]['comb'] = {}
        profdat[im]['comb']['profcb'] = profc
        profdat[im]['comb']['profcbsub'] = profcsub

        profdat[im]['comb']['cov_scaling'] = covc_scaling
        profdat[im]['comb']['covsub_scaling'] = covcsub_scaling
        profdat[im]['comb']['cov_stack'] = covc_stack
        profdat[im]['comb']['covsub_stack'] = covcsub_stack
        profdat[im]['comb']['cov_gaia_sys'] = cov_gaia_sys
        profdat[im]['comb']['covsub_gaia_sys'] = covsub_gaia_sys
        profdat[im]['comb']['cov'] = profdat[im]['comb']['cov_gaia_sys'] + profdat[im]['comb']['cov_scaling']
        profdat[im]['comb']['covsub'] = profdat[im]['comb']['covsub_gaia_sys'] + profdat[im]['comb']['covsub_scaling']
        
        profdat[im]['comb']['profcb_err_scaling'] = np.sqrt(np.diag(profdat[im]['comb']['cov_scaling']))
        profdat[im]['comb']['profcbsub_err_scaling'] = np.sqrt(np.diag(profdat[im]['comb']['covsub_scaling']))
        profdat[im]['comb']['profcb_err_stack'] = np.sqrt(np.diag(profdat[im]['comb']['cov_stack']))
        profdat[im]['comb']['profcbsub_err_stack'] = np.sqrt(np.diag(profdat[im]['comb']['covsub_stack']))
        profdat[im]['comb']['profcb_err_gaia_sys'] = np.sqrt(np.diag(profdat[im]['comb']['cov_gaia_sys']))
        profdat[im]['comb']['profcbsub_err_gaia_sys'] = np.sqrt(np.diag(profdat[im]['comb']['covsub_gaia_sys']))
        profdat[im]['comb']['profcb_err'] = np.sqrt(np.diag(profdat[im]['comb']['cov']))
        profdat[im]['comb']['profcbsub_err'] = np.sqrt(np.diag(profdat[im]['comb']['covsub']))
        
        profdat[im]['comb']['cov_rho'] = normalize_cov(profdat[im]['comb']['cov'])
        profdat[im]['comb']['covsub_rho'] = normalize_cov(profdat[im]['comb']['covsub'])
        profdat[im]['comb']['log_slopes'] = (slope_mid, slope_out)
        profdat[im]['comb']['r_connect'] = (profdat['rbins'][11],profdat['rbins'][13])
        profdat[im]['comb']['r_connect_idx'] = (11,13)
        profdat[im]['comb']['rsub_connect_idx'] = (6,8)

    if savedata:
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)
    
    return profdat

def psf_comb_interpolate(inst, ifield, im, r_arr):
    r_arr = np.array(r_arr)
    psf_interp = np.zeros_like(r_arr, dtype=float)
    
    fname = mypaths['alldat'] + 'TM'+ str(inst) +\
     '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    with open(fname,"rb") as f:
        profdat = pickle.load(f)

    r1,r2 = profdat[im]['comb']['r_connect']
    idx1, idx2 = profdat[im]['comb']['r_connect_idx']
    rbins = profdat['rbins']
    prof = profdat[im]['comb']['profcb']
    slope_mid, slope_out = profdat[im]['comb']['log_slopes']
    
    tck = interpolate.splrep(np.log(rbins), np.log(prof), s=0)
    psf_interp[(r_arr<=r1) & (r_arr>0)] = np.exp(interpolate.splev\
                                                 (np.log(r_arr[(r_arr<=r1) & (r_arr>0)]), tck, der=0))
    
    norm_mid = prof[idx1] / 10**(slope_mid * np.log10(rbins[idx1]))
    psf_interp[(r_arr>r1) & (r_arr<=r2)] = 10**(slope_mid * np.log10(r_arr[(r_arr>r1)\
                                                   & (r_arr<=r2)])) * norm_mid
    norm_out = prof[idx2] / 10**(slope_out * np.log10(rbins[idx2]))
    psf_interp[r_arr>r2] = 10**(slope_out * np.log10(r_arr[r_arr>r2])) * norm_out
    
    psf_interp[r_arr==0] = np.max(psf_interp[r_arr!=0])
    
    return psf_interp

def run_psf_combine_old(inst, ifield, savedata=True, idx_comb=(9,10)):
    
    fname = mypaths['alldat'] + 'TM'+ str(inst) + \
    '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    with open(fname, "rb") as f:
        profdat = pickle.load(f)

    i1, i2 = idx_comb

    for im,(m_min,m_max) in enumerate(zip(magbindict['m_min'],magbindict['m_max'])):

        profc = np.zeros_like(profdat['out']['profcb'])
        profcsub = np.zeros_like(profdat['out']['profcbsub'])
        covc = np.zeros_like(profdat['out']['cov'])
        covcsub = np.zeros_like(profdat['out']['covsub'])

        if ifield==5 and im==3:
            prof_in = profdat[2]['profcb'].copy()
            prof_in_sub = profdat[2]['profcbsub'].copy()
            prof_in_cov = profdat[2]['cov'].copy()
            prof_in_cov_sub = profdat[2]['covsub'].copy()
        else:
            prof_in = profdat[im]['profcb'].copy()
            prof_in_sub = profdat[im]['profcbsub'].copy()
            prof_in_cov = profdat[im]['cov'].copy()
            prof_in_cov_sub = profdat[im]['covsub'].copy()

        norm_in = 1 / prof_in[0]
        norm_mid = norm_in * prof_in[i1] / profdat['mid']['profcb'][i1]
        norm_out = norm_mid * profdat['mid']['profcb'][i2] / profdat['out']['profcb'][i2]

        profc[:i1+1] = prof_in[:i1+1].copy() * norm_in
        profc[i1+1:i2+1] = profdat['mid']['profcb'][i1+1:i2+1].copy() * norm_mid
        profc[i2+1:] = profdat['out']['profcb'][i2+1:].copy() * norm_out

        profcsub[:4] = prof_in_sub[:4].copy() * norm_in
        profcsub[4:6] = profdat['mid']['profcbsub'][4:6].copy() * norm_mid
        profcsub[6:] = profdat['out']['profcbsub'][6:].copy() * norm_out

        covc[:i1+1,:i1+1] = prof_in_cov[:i1+1,:i1+1].copy() * norm_in**2
        covc[i1+1:i2+1,i1+1:i2+1] = profdat['mid']['cov'][i1+1:i2+1,10:i2+1].copy() * norm_mid**2
        covc[i2+1:,i2+1:] = profdat['out']['cov'][i2+1:,i2+1:].copy() * norm_out**2

        covc_rho = np.zeros_like(covc)
        for i in range(covc_rho.shape[0]):
            for j in range(covc_rho.shape[0]):
                if covc[i,i]==0 or covc[j,j]==0:
                    covc_rho[i,j] = covc[i,j]
                else:
                    covc_rho[i,j] = covc[i,j] / np.sqrt(covc[i,i]*covc[j,j])

        covcsub[:4,:4] = prof_in_cov_sub[:4,:4].copy() * norm_in**2
        covcsub[4:6,4:6] = profdat['mid']['covsub'][4:6,4:6].copy() * norm_mid**2
        covcsub[6:,6:] = profdat['out']['covsub'][6:,6:].copy() * norm_out**2

        covcsub_rho = np.zeros_like(covcsub)
        for i in range(covcsub_rho.shape[0]):
            for j in range(covcsub_rho.shape[0]):
                if covcsub[i,i]==0 or covcsub[j,j]==0:
                    covcsub_rho[i,j] = covcsub[i,j]
                else:
                    covcsub_rho[i,j] = covcsub[i,j] / np.sqrt(covcsub[i,i]*covcsub[j,j])

        profdat[im]['comb'] = {}
        profdat[im]['comb']['profcb'] = profc
        profdat[im]['comb']['profcb_err'] = np.sqrt(np.diag(covc))
        profdat[im]['comb']['profcbsub'] = profcsub
        profdat[im]['comb']['profcbsub_err'] = np.sqrt(np.diag(covcsub))
        profdat[im]['comb']['cov'] = covc
        profdat[im]['comb']['covsub'] = covcsub
        profdat[im]['comb']['cov_rho'] = covc_rho
        profdat[im]['comb']['covsub_rho'] = covcsub_rho

    if savedata:
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)
    
    return profdat

def run_psf_synth_2m_mag(inst, ifield, m_min, m_max, data_maps=None,
    filt_order=3, savedata=True):
    
    if data_maps is None:
        data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    cal = -cal_factor_dict['apf2nWpm2psr'][inst][ifield]
    
    psfdata_out = stack_psf(inst, data_maps[inst].stackmapdat,m_min=m_min, m_max=m_max,
     ifield_arr=[ifield], Nsub_single=True, savedata=False, save_stackmap=False)

    profdat = {}
    profdat['rbins'] = psfdata_out[ifield]['rbins']
    profdat['rbinedges'] = psfdata_out[ifield]['rbinedges']
    profdat['rsubbins'] = psfdata_out[ifield]['rsubbins']
    profdat['rsubbinedges'] = psfdata_out[ifield]['rsubbinedges']
    profdat['filt_order'] = filt_order

    profdat['m_min'] = m_min
    profdat['m_max'] = m_max
    profdat['Nsrc'] = psfdata_out[ifield]['Nsrc']
    profdat['profcb'] = psfdata_out[ifield]['prof']*cal
    profdat['profcb_err'] = psfdata_out[ifield]['prof_err']*cal
    profdat['profcbsub'] = psfdata_out[ifield]['profsub']*cal
    profdat['profcbsub_err'] = psfdata_out[ifield]['profsub_err']*cal
    profdat['cov'] = psfdata_out[ifield]['cov']*cal**2
    profdat['covsub'] = psfdata_out[ifield]['covsub']*cal**2
    
    if savedata:
        fname = mypaths['alldat'] + 'TM'+ str(inst) +\
         '/psfdata_synth_2m_%s_%d_%d.pkl'%(fieldnamedict[ifield],m_min, m_max)
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)

        return profdat

    return profdat

def run_psf_synth_ps_mag(inst, ifield, m_min, m_max, data_maps=None,
 filt_order=3, savedata=True, ):

    fname = mypaths['alldat'] + 'TM'+ str(inst) + \
    '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    with open(fname, "rb") as f:
        profdat0 = pickle.load(f)

    if data_maps is None:
        data_maps = {1: image_reduction(1), 2: image_reduction(2)}
    
    profdat = {}
    profdat['rbins'] = profdat0['rbins']
    profdat['rbinedges'] = profdat0['rbinedges']
    profdat['rsubbins'] = profdat0['rsubbins']
    profdat['rsubbinedges'] = profdat0['rsubbinedges']
    profdat['filt_order'] = filt_order
    
    mapin, strmask, strnum, mask_inst1, mask_inst2 = \
    load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                       (inst,ifield,'strmask'), 
                                       (inst,ifield,'strnum'),
                                       (1,ifield,'mask_inst'),
                                       (2,ifield,'mask_inst')])
    
    stack_class = stacking(inst, ifield, 16, 17, filt_order=filt_order, 
                        load_from_file=True,BGsub=False)

    stack_class.m_min = m_min
    stack_class.m_max = m_max
    cliplim = stack_class._stackihl_PS_cliplim()

    srcdat = ps_src_select(inst, ifield, m_min, m_max, 
        [mask_inst1, mask_inst2], sample_type='jack_region')
    if srcdat['Ns'] < srcdat['Nsub']:
        srcdat = ps_src_select(inst, ifield, m_min, m_max, 
            [mask_inst1, mask_inst2], sample_type='jack_random',
            Nsub=srcdat['Ns'])                

    stackdat = stack_class.stack_PS(srctype='s',cliplim=cliplim, 
                                    srcdat=srcdat, verbose=False)
    stack_class.stackdat = stackdat
    stack_class._get_jackknife_profile()
    stack_class._get_covariance()

    profdat['m_min'] = m_min
    profdat['m_max'] = m_max
    profdat['Nsrc'] = stackdat['Nsrc']
    profdat['profcb'] = stack_class.stackdat['profcb']
    profdat['profcb_err'] = np.sqrt(np.diag(stackdat['cov']['profcb']))
    profdat['profcbsub'] = stack_class.stackdat['profcbsub']
    profdat['profcbsub_err'] = np.sqrt(np.diag(stackdat['cov']['profcbsub']))
    profdat['cov'] = stackdat['cov']['profcb']
    profdat['covsub'] = stackdat['cov']['profcbsub']
    profdat['sub'] = stackdat['sub']

    if savedata:
        fname = mypaths['alldat'] + 'TM'+ str(inst) +\
         '/psfdata_synth_ps_%s_%d_%d.pkl'%(fieldnamedict[ifield],m_min, m_max)
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)
        
    return profdat

def stack_gaia(inst, ifield, data_maps=None, m_min=12, m_max=14, 
    target_filter=None, Nsub=10,
    filt_order=3, Nsub_single=False, save_stackmap=False, 
    savedata=True, savename=None):

    if data_maps is None:
        data_maps = {1: image_reduction(1), 2: image_reduction(2)}
        
    cbmap, mask_inst= \
    load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                       (inst,ifield,'mask_inst')])
    
    # get data & mask
    catdir = mypaths['GAIAcatdat']
    # df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')
    
    df = pd.read_csv(catdir + fieldnamedict[ifield] + '_raw.csv')#####
    # df = catalog_add_xy_from_radec(fieldnamedict[ifield], df)####
    
    xs = df['y'+str(inst)].values
    ys = df['x'+str(inst)].values
    ms = df['phot_g_mean_mag'].values
    sp = np.where((ms < 21) & (xs>-20) & (xs<1044) & (ys>-20) & (ys<1044))[0]
    xs, ys, ms = xs[sp], ys[sp], ms[sp]
    rs = get_mask_radius_th(ifield, ms-1)
    
    strmask = np.ones([1024,1024])
    strnum = np.zeros([1024,1024])
    for i,(x,y,r) in enumerate(zip(xs, ys, rs)):
        radmap = make_radius_map(strmask, x, y)
        strmask[radmap < r/7.] = 0
        strnum[radmap < r/7.] += 1

    Q1 = np.percentile(cbmap[(mask_inst*strmask==1)], 25)
    Q3 = np.percentile(cbmap[(mask_inst*strmask==1)], 75)
    clipmin = Q1 - 3 * (Q3 - Q1)
    clipmax = Q3 + 3 * (Q3 - Q1)
    mask_inst[(cbmap > clipmax) & (mask_inst*strmask==1)] = 0
    mask_inst[(cbmap < clipmin) & (mask_inst*strmask==1)] = 0
    cbmap = cbmap - np.mean(cbmap[mask_inst*strmask==1])

    # get cliplim
    df = df[df['parallax']==df['parallax']]
    xs = df['y'+str(inst)].values
    ys = df['x'+str(inst)].values
    ms = df['phot_g_mean_mag'].values
    parallax = df['parallax'].values
    sp = np.where((ms>m_min) & (ms<m_max) &\
     (xs>-0.5) & (xs<1023.5) & (ys>-0.5) & (ys<1023.5) &\
      (parallax > 1/5e3))[0]
    xs, ys, ms = xs[sp], ys[sp], ms[sp]
    
    if target_filter is not None:
        if target_filter == 'parallax_over_error':
            poe = df['parallax_over_error'].values
            poe = poe[sp]
            sp = np.where(poe > 2)[0]
            xs, ys, ms = xs[sp], ys[sp], ms[sp]
        elif target_filter == 'astrometric_excess_noise':
            aen = df['astrometric_excess_noise'].values
            aen = aen[sp]
            sp = np.where(aen == 0)[0]
            xs, ys, ms = xs[sp], ys[sp], ms[sp]
        elif target_filter == 'astrometric_gof_al':
            aga = df['astrometric_gof_al'].values
            aga = aga[sp]
            sp = np.where(aga < 3)[0]
            xs, ys, ms = xs[sp], ys[sp], ms[sp]

    rs = get_mask_radius_th(ifield, ms-1)
    
    nbins = 25
    dx = 1200
    profile = radial_prof(np.ones([2*dx+1,2*dx+1]), dx, dx)
    rbinedges, rbins = profile['rbinedges'], profile['rbins'] # subpix units
    cliplim = {'rbins': rbins*0.7, 'rbinedges': rbinedges*0.7,
              'CBmax': np.full((nbins), np.inf),
              'CBmin': np.full((nbins), -np.inf),
              }
    if m_max <= 21:
        if len(ms)>1000:
            sp = np.arange(len(ms))
            np.random.shuffle(sp)
            sp = sp[:1000]
        else:
            sp = np.arange(len(ms))
        x_arr, y_arr, m_arr, r_arr = xs[sp], ys[sp], ms[sp], rs[sp]


        cbdata = {}
        for i in range(len(rbins)):
            cbdata[i] = np.array([])

        for isrc in range(len(x_arr)):
            radmap = make_radius_map(cbmap, x_arr[isrc], y_arr[isrc]) # large pix units

            sp1 = np.where((radmap < r_arr[isrc]/7) & (strnum==1) & (mask_inst==1))
            if len(sp1[0])==0:
                continue
            # unmasked radii and their CBmap values
            ri = radmap[sp1]*10 # sub pix units
            cbi = cbmap[sp1]

            for ibin in range(len(rbins)):
                spi = np.where((ri>rbinedges[ibin]) & (ri<rbinedges[ibin+1]))[0]
                if len(spi)==0:
                    continue
                cbdata[ibin] = np.append(cbdata[ibin], cbi[spi])


        d = np.concatenate((cbdata[0],cbdata[1],cbdata[2],cbdata[3]))
        Q1, Q3 = np.percentile(d, 25), np.percentile(d, 75)
        IQR = Q3 - Q1
        cliplim['CBmin'][:4], cliplim['CBmax'][:4]= Q1-3*IQR, Q3+3*IQR
    
        for ibin in np.arange(4,nbins,1):
            d = cbdata[ibin]
            if len(d)==0:
                continue
            Q1, Q3 = np.percentile(d, 25), np.percentile(d, 75)
            IQR = Q3 - Q1
            cliplim['CBmin'][ibin], cliplim['CBmax'][ibin]= Q1-3*IQR, Q3+3*IQR
    
    # stack
    stack_class = stacking_mock(inst)
    psfdata = {}

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
        = stack_class.run_stacking(cbmap, mask_inst*strmask, strnum, 
                                   mask_inst=mask_inst,return_all=True,
                                update_mask=False,cliplim=cliplim, verbose=True)
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

    profdat = {}
    profdat['Nsrc'] = len(xs)
    profdat['m_min'] = m_min
    profdat['m_max'] = m_max
    profdat['rbins'] = stackdat['rbins'].copy()
    profdat['rbinedges'] = stackdat['rbinedges'].copy()
    profdat['rsubbins'] = stackdat['rsubbins'].copy()
    profdat['rsubbinedges'] = stackdat['rsubbinedges'].copy()
    profdat['profcb'] = prof
    profdat['profcbsub'] = profsub
    profdat['profhit'] = np.sum(profhit_arr,axis=0)
    profdat['profcb_err'] = np.sqrt(np.diag(cov))
    profdat['profcbsub_err'] = np.sqrt(np.diag(covsub))
    profdat['cov'] = cov
    profdat['covsub'] = covsub
    if save_stackmap:
        profdat['stackmap'] = stack

    if savedata:
        if savename is None:
            savename = 'psfdata_synth_gaia_%s_%d_%d.pkl'%(fieldnamedict[ifield],m_min, m_max)
        fname = mypaths['alldat'] + 'TM'+ str(inst) + '/'+ savename
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)

    return profdat

def run_psf_synth_mag_all(inst, ifield):

    data_maps = {1: image_reduction(1), 2: image_reduction(2)}
    filt_order = filt_order_dict[inst]
    run_psf_synth_2m_mag(inst, ifield, 4, 9, filt_order=filt_order,
     data_maps=data_maps)
    run_psf_synth_2m_mag(inst, ifield, 9, 10, filt_order=filt_order,
        data_maps=data_maps)
    run_psf_synth_ps_mag(inst, ifield, 12, 13, filt_order=filt_order,
        data_maps=data_maps)
    run_psf_synth_ps_mag(inst, ifield, 13, 14, filt_order=filt_order,
        data_maps=data_maps)
    run_psf_synth_ps_mag(inst, ifield, 14, 15, filt_order=filt_order,
        data_maps=data_maps)
    run_psf_synth_ps_mag(inst, ifield, 15, 16, filt_order=filt_order,
        data_maps=data_maps)

    return

def run_psf_synth_mag_all_gaia(inst, ifield, m_min_arr, m_max_arr):

    data_maps = {1: image_reduction(1), 2: image_reduction(2)}
    filt_order = filt_order_dict[inst]
    for m_min, m_max in zip(m_min_arr, m_max_arr):
        stack_gaia(inst, ifield, data_maps=data_maps, m_min=m_min, m_max=m_max,
            filt_order=filt_order)

    return

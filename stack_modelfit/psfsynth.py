from reduction import *
from stack import * 
from psfstack import *

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

def run_psf_combine(inst, ifield):
    
    fname = mypaths['alldat'] + 'TM'+ str(inst) + \
    '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    with open(fname, "rb") as f:
        profdat = pickle.load(f)

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
        norm_mid = norm_in * prof_in[8] / profdat['mid']['profcb'][8]
        norm_out = norm_mid * profdat['mid']['profcb'][10] / profdat['out']['profcb'][10]

        profc[:9] = prof_in[:9].copy() * norm_in
        profc[9:11] = profdat['mid']['profcb'][9:11].copy() * norm_mid
        profc[11:] = profdat['out']['profcb'][11:].copy() * norm_out

        profcsub[:4] = prof_in_sub[:4].copy() * norm_in
        profcsub[4:6] = profdat['mid']['profcbsub'][4:6].copy() * norm_mid
        profcsub[6:] = profdat['out']['profcbsub'][6:].copy() * norm_out

        covc[:9,:9] = prof_in_cov[:9,:9].copy() * norm_in**2
        covc[9:11,9:11] = profdat['mid']['cov'][9:11,9:11].copy() * norm_mid**2
        covc[11:,11:] = profdat['out']['cov'][11:,11:].copy() * norm_out**2

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

    with open(fname, "wb") as f:
        pickle.dump(profdat, f)
    
    return profdat



def run_psf_synth_temp(inst, ifield, filt_order=3, savedata=True):

    fname = mypaths['alldat'] + 'TM'+ str(inst) + \
    '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    with open(fname, "rb") as f:
        profdat = pickle.load(f)

    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

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
        # if im !=3:
        #     continue
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
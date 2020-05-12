from reduction import *
from stack import * 
from psfstack import *

def run_psf_synth(inst, ifield, filt_order=3, savedata=True):

    fname = mypaths['alldat'] + 'TM'+ str(inst) + \
    '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    with open(fname, "rb") as f:
        profdat = pickle.load(f)

    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    psfdata_out = stack_psf(inst, data_maps[inst].stackmapdat,m_min=4, m_max=9,
     ifield_arr=[ifield], Nsub_single=True, savedata=False, save_stackmap=False)

    # profdat = {}
    # profdat['rbins'] = psfdata_out[ifield]['rbins']
    # profdat['rbinedges'] = psfdata_out[ifield]['rbinedges']
    # profdat['rsubbins'] = psfdata_out[ifield]['rsubbins']
    # profdat['rsubbinedges'] = psfdata_out[ifield]['rsubbinedges']
    # profdat['filt_order'] = filt_order

    profdat['out'] = {}
    profdat['out']['m_min'] = 4
    profdat['out']['m_max'] = 9
    profdat['out']['Nsrc'] = psfdata_out[ifield]['Nsrc']
    profdat['out']['profcb'] = psfdata_out[ifield]['prof']
    profdat['out']['profcb_err'] = psfdata_out[ifield]['prof_err']
    profdat['out']['profcbsub'] = psfdata_out[ifield]['profsub']
    profdat['out']['profcbsub_err'] = psfdata_out[ifield]['profsub_err']
    profdat['out']['cov'] = psfdata_out[ifield]['cov']
    profdat['out']['covsub'] = psfdata_out[ifield]['covsub']

    psfdata_mid = stack_psf(inst, data_maps[inst].stackmapdat, m_min=13, m_max=14,
     ifield_arr=[ifield], Nsub_single=True, savedata=False, save_stackmap=False)

    profdat['mid'] = {}
    profdat['mid']['m_min'] = 13
    profdat['mid']['m_max'] = 14
    profdat['mid']['Nsrc'] = psfdata_mid[ifield]['Nsrc']
    profdat['mid']['profcb'] = psfdata_mid[ifield]['prof']
    profdat['mid']['profcb_err'] = psfdata_mid[ifield]['prof_err']
    profdat['mid']['profcbsub'] = psfdata_mid[ifield]['profsub']
    profdat['mid']['profcbsub_err'] = psfdata_mid[ifield]['profsub_err']
    profdat['mid']['cov'] = psfdata_mid[ifield]['cov']
    profdat['mid']['covsub'] = psfdata_mid[ifield]['covsub']

    if savedata:
        fname = mypaths['alldat'] + 'TM'+ str(inst) +\
         '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)

    # mapin, strmask, strnum, mask_inst1, mask_inst2 = \
    # load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
    #                                    (inst,ifield,'strmask'), 
    #                                    (inst,ifield,'strnum'),
    #                                    (1,ifield,'mask_inst'),
    #                                    (2,ifield,'mask_inst')])
    
    # for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):

    #     stack_class = stacking(inst, ifield, m_min, m_max, filt_order=filt_order, 
    #                         load_from_file=True,BGsub=False)

    #     cliplim = stack_class._stackihl_PS_cliplim()

    #     srcdat = ps_src_select(inst, ifield, m_min, m_max, 
    #         [mask_inst1, mask_inst2], sample_type='jack_region')

    #     stackdat = stack_class.stack_PS(srctype='s',cliplim=cliplim, 
    #                                     srcdat=srcdat, verbose=False)
    #     stack_class.stackdat = stackdat
    #     stack_class._get_jackknife_profile()
    #     stack_class._get_covariance()

    #     profdat[im] = {}
    #     profdat[im]['m_min'] = m_min
    #     profdat[im]['m_max'] = m_max
    #     profdat[im]['Nsrc'] = stackdat['Nsrc']
    #     profdat[im]['profcb'] = stack_class.stackdat['profcb']
    #     profdat[im]['profcb_err'] = np.sqrt(np.diag(stackdat['cov']['profcb']))
    #     profdat[im]['profcbsub'] = stack_class.stackdat['profcbsub']
    #     profdat[im]['profcbsub_err'] = np.sqrt(np.diag(stackdat['cov']['profcbsub']))
    #     profdat[im]['cov'] = stackdat['cov']['profcb']
    #     profdat[im]['covsub'] = stackdat['cov']['profcbsub']

    #     if savedata:
    #         fname = mypaths['alldat'] + 'TM'+ str(inst) +\
    #          '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
    #         with open(fname, "wb") as f:
    #             pickle.dump(profdat, f)
        
    if savedata:
        return
    
    return profdat
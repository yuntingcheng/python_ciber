from reduction import *
from stack import * 
from psfstack import *

def run_psf_synth(inst, ifield, filt_order=3, savedata=True):
#     fname = mypaths['alldat'] + 'TM'+ str(inst) + '/psfdata_synth.pkl'###
#     with open(fname,"rb") as f:###
#         profdat1 = pickle.load(f)###
    
    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    # psfdata_in = stack_psf(inst, data_maps[inst].stackmapdat,m_min=4, m_max=9,
    #  Nsub_single=True, savedata=False, save_stackmap=False)
    psfdata_in = stack_psf(inst, data_maps[inst].stackmapdat,m_min=4, m_max=9,
     ifield=ifield, Nsub_single=True, savedata=False, save_stackmap=False)

    profdat = {}
    profdat['rbins'] = psfdata_in[ifield]['rbins']
    profdat['rbinedges'] = psfdata_in[ifield]['rbinedges']
    profdat['rsubbins'] = psfdata_in[ifield]['rsubbins']
    profdat['rsubbinedges'] = psfdata_in[ifield]['rsubbinedges']
    profdat['filt_order'] = filt_order

    profdat['in'] = {}
    profdat['in']['m_min'] = 4
    profdat['in']['m_max'] = 9
    profdat['in']['Nsrc'] = psfdata_in[ifield]['Nsrc']
    profdat['in']['profcb'] = psfdata_in[ifield]['prof']
    profdat['in']['profcb_err'] = psfdata_in[ifield]['prof_err']
    profdat['in']['profcbsub'] = psfdata_in[ifield]['profsub']
    profdat['in']['profcbsub_err'] = psfdata_in[ifield]['profsub_err']
    profdat['in']['cov'] = psfdata_in[ifield]['cov']
    profdat['in']['covsub'] = psfdata_in[ifield]['covsub']

    # psfdata_mid = stack_psf(inst, data_maps[inst].stackmapdat, m_min=13, m_max=14,
    #  Nsub_single=True, savedata=False, save_stackmap=False)
    psfdata_mid = stack_psf(inst, data_maps[inst].stackmapdat, m_min=4, m_max=9,
     ifield=ifield, Nsub_single=True, savedata=False, save_stackmap=False)

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

    mapin, strmask, strnum, mask_inst1, mask_inst2 = \
    load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                       (inst,ifield,'strmask'), 
                                       (inst,ifield,'strnum'),
                                       (1,ifield,'mask_inst'),
                                       (2,ifield,'mask_inst')])
    
    # for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):
    for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'][:1], magbindict['m_max'][:1])):

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

        if savedata:
            fname = mypaths['alldat'] + 'TM'+ str(inst) +\
             '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
            with open(fname, "wb") as f:
                pickle.dump(profdat, f)
        
    if savedata:
        return
    
    return profdat
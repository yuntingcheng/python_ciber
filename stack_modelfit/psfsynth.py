from reduction import *
from stack import * 
from psfstack import *

def run_psf_synth(inst, ifield, savedata=True):
#     fname = mypaths['alldat'] + 'TM'+ str(inst) + '/psfdata_synth.pkl'###
#     with open(fname,"rb") as f:###
#         profdat1 = pickle.load(f)###
    
    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

psfdata_in = stack_psf(inst, data_maps[inst].stackmapdat,
                m_min=4, m_max=9, Nsub_single=True, savedata=False)
psfdata_mid = stack_psf(inst, data_maps[inst].stackmapdat,
                m_min=13, m_max=14, Nsub_single=True, savedata=False)

    mapin, strmask, strnum, mask_inst1, mask_inst2 = \
    load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                       (inst,ifield,'strmask'), 
                                       (inst,ifield,'strnum'),
                                       (1,ifield,'mask_inst'),
                                       (2,ifield,'mask_inst')])

    profdat = {}
    profdat['rbins'] = psfdata_in[ifield]['rbins']
    profdat['rbinedges'] = psfdata_in[ifield]['rbinedges']
    profdat['rsubbins'] = psfdata_in[ifield]['rsubbins']
    profdat['rsubbinedges'] = psfdata_in[ifield]['rsubbinedges']

    profdat['in'] = {}
    profdat['in']['m_min'] = 4
    profdat['in']['m_max'] = 9
    profdat['in']['profcb'] = psfdata_in[ifield]['prof']
    profdat['in']['profcb_err'] = psfdata_in[ifield]['prof_err']
    profdat['in']['profcbsub'] = psfdata_in[ifield]['profsub']
    profdat['in']['profcbsub_err'] = psfdata_in[ifield]['profsub_err']

    profdat['mid'] = {}
    profdat['mid']['m_min'] = 13
    profdat['mid']['m_max'] = 14
    profdat['mid']['profcb'] = psfdata_mid[ifield]['prof']
    profdat['mid']['profcb_err'] = psfdata_mid[ifield]['prof_err']
    profdat['mid']['profcbsub'] = psfdata_mid[ifield]['profsub']
    profdat['mid']['profcbsub_err'] = psfdata_mid[ifield]['profsub_err']

    if savedata:
        fname = mypaths['alldat'] + 'TM'+ str(inst) +\
         '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)
    
    for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):
        stack_class = stacking(inst, ifield, m_min, m_max, 
                            load_from_file=True,BGsub=False)

        cliplim = stack_class._stackihl_PS_cliplim()

        srcdat = ps_src_select(inst, ifield, m_min, m_max, 
            [mask_inst1, mask_inst2], sample_type='jack_random', Nsub=10, Nsrc_use=None)

        stackdat = stack_class.stack_PS(srctype='s', sample_type='jack_region',
                                      cliplim=cliplim, srcdat=srcdat)
        stack_class.stackdat = stackdat
        stack_class._get_jackknife_profile()
        stack_class._get_covariance()

        profdat['rsubbins'] = stack_class.stackdat['rsubbins']
        profdat['rsubbinedges'] = stack_class.stackdat['rsubbinedges']

        profdat[im] = {}
        profdat[im]['m_min'] = m_min
        profdat[im]['m_max'] = m_max
        profdat[im]['profcb'] = stack_class.stackdat['profcb']
        profdat[im]['profcbsub'] = stack_class.stackdat['profcbsub']
        profdat[im]['profcb_err'] = np.sqrt(np.diag(stackdat['cov']['profcb']))
        profdat[im]['profcbsub_err'] = np.sqrt(np.diag(stackdat['cov']['profcbsub']))
#         profdat[im] = profdat1[ifield][im]
    
        if savedata:
            fname = mypaths['alldat'] + 'TM'+ str(inst) +\
             '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
            with open(fname, "wb") as f:
                pickle.dump(profdat, f)
        
    if savedata:
        return
    
    return profdat
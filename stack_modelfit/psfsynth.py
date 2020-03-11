from reduction import *
from stack import * 
from psfstack import *

def run_psf_synth(inst, ifield, savedata=True):
    
    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    psfdata = stack_psf(inst, data_maps[inst].stackmapdat,
                    m_min=4, m_max=9, Nsub_single=True, savedata=False)

    mapin, strmask, strnum, mask_inst1, mask_inst2 = \
    load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                       (inst,ifield,'strmask'), 
                                       (inst,ifield,'strnum'),
                                       (1,ifield,'mask_inst'),
                                       (2,ifield,'mask_inst')])

    profdat = {}
    profdat['m_min'] = 4
    profdat['m_max'] = 9
    profdat['rbins'] = psfdata[ifield]['rbins']
    profdat['rbinedges'] = psfdata[ifield]['rbinedges']
    profdat['profcb'] = psfdata[ifield]['prof']
    profdat['profcb_err'] = psfdata[ifield]['prof_err']

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
        profdat[im]['m_max'] = m_min
        profdat[im]['profcb'] = stack_class.stackdat['profcb']
        profdat[im]['profcbsub'] = stack_class.stackdat['profcbsub']
        profdat[im]['profcb_err'] = np.sqrt(np.diag(stackdat['cov']['profcb']))
        profdat[im]['profcbsub_err'] = np.sqrt(np.diag(stackdat['cov']['profcbsub']))

        if savedata:
            fname = mypaths['alldat'] + 'TM'+ str(inst) +\
             '/psfdata_synth_%s.pkl'%(fieldnamedict[ifield])
            with open(fname, "wb") as f:
                pickle.dump(profdat, f)
        
    if savedata:
        return
    
    return profdat
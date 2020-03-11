from reduction import *
from stack import * 

def run_psf_synth(inst, savedata=True):
    
    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    psfdata = stack_psf(inst, data_maps[inst].stackmapdat,
                    m_min=4, m_max=9, Nsub=None, savedata=False)
    profdat = {}
    for ifield in [4,5,6,7,8]:
        profdat[ifield] = {}
        mapin, strmask, strnum, mask_inst1, mask_inst2 = \
        load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                           (inst,ifield,'strmask'), 
                                           (inst,ifield,'strnum'),
                                           (1,ifield,'mask_inst'),
                                           (2,ifield,'mask_inst')])

        profdat[ifield][0] = {}
        profdat[ifield][0]['m_min'] = 4
        profdat[ifield][0]['m_max'] = 9
        profdat[ifield][0]['rbins'] = psfdata[ifield]['rbins']
        profdat[ifield][0]['rbinedges'] = psfdata[ifield]['rbinedges']
        profdat[ifield][0]['profcb'] = psfdata[ifield]['prof']
        profdat[ifield][0]['profcb_err'] = psfdata[ifield]['prof_err']
    
    for ifield in [4,5,6,7,8]:
        for im, (m_min, m_max) in enumerate(zip(magbindict['m_min'], magbindict['m_max'])):
            stack_class = stacking(inst, ifield, m_min, m_max, 
                                load_from_file=True,BGsub=False)

            cliplim = stack_class._stackihl_PS_cliplim()

            srcdat = ps_src_select(inst, ifield, m_min, m_max, 
                [mask_inst1, mask_inst2], sample_type='jack_random', Nsub=10, Nsrc_use=100)

            stackdat = stack_class.stack_PS(srctype='s', sample_type='jack_region',
                                          cliplim=cliplim, srcdat=srcdat)
            stack_class.stackdat = stackdat
            stack_class._get_jackknife_profile()
            stack_class._get_covariance()

            profdat[ifield]['rbins'] = stack_class.stackdat['rbins']
            profdat[ifield]['rbinedges'] = stack_class.stackdat['rbinedges']
            profdat[ifield]['rsubbins'] = stack_class.stackdat['rsubbins']
            profdat[ifield]['rsubbinedges'] = stack_class.stackdat['rsubbinedges']

            profdat[ifield][im] = {}
            profdat[ifield][im]['m_min'] = m_min
            profdat[ifield][im]['m_max'] = m_min
            profdat[ifield][im]['profcb'] = stack_class.stackdat['profcb']
            profdat[ifield][im]['profcbsub'] = stack_class.stackdat['profcbsub']
            profdat[ifield][im]['profcb_err'] = np.sqrt(np.diag(stackdat['cov']['profcb']))
            profdat[ifield][im]['profcbsub_err'] = np.sqrt(np.diag(stackdat['cov']['profcbsub']))

    if savedata:
        fname = mypaths['alldat'] + 'TM'+ str(inst) + '/psfdata_synth.pkl'
        with open(fname, "wb") as f:
            pickle.dump(profdat, f)
        return
    
    return profdat
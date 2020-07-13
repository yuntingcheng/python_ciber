from reduction import *
from psfsynth import *

def run_gaia_test_stack(m_min, m_max):
    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    inst = 1
    filt_order = filt_order_dict[inst]
    ifield = 6
    
    savename='psfdata_synth_gaia_%s_%d_%d_g.pkl'%(fieldnamedict[ifield],m_min, m_max)
    profdatg = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_gof_al',
                          m_min=m_min, m_max=m_max, filt_order=filt_order,
                           savename=savename)

    savename='psfdata_synth_gaia_%s_%d_%d_e.pkl'%(fieldnamedict[ifield],m_min, m_max)
    profdate = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_excess_noise',
                          m_min=m_min, m_max=m_max, filt_order=filt_order,
                           savename=savename)

    savename='psfdata_synth_gaia_%s_%d_%d_p.pkl'%(fieldnamedict[ifield],m_min, m_max)
    profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='parallax_over_error',
                          m_min=m_min, m_max=m_max, filt_order=filt_order,
                           savename=savename)
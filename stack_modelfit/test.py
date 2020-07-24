from reduction import *
from psfsynth import *

def run_gaia_test_stack(inst,ifield,m_min, m_max, target_filter=None, **kwargs):
    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    filt_order = filt_order_dict[inst]
    
    if target_filter is None:
        savename='psfdata_synth_gaia_%s_%d_%d_g.pkl'%(fieldnamedict[ifield],m_min, m_max)
        profdatg = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_gof_al',
                              m_min=m_min, m_max=m_max, filt_order=filt_order,
                               savename=savename)

        savename='psfdata_synth_gaia_%s_%d_%d_e.pkl'%(fieldnamedict[ifield],m_min, m_max)
        profdate = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_excess_noise',
                              m_min=m_min, m_max=m_max, filt_order=filt_order,
                               savename=savename)

        if 'parallax_over_error_th' in kwargs.keys():
            parallax_over_error_th = kwargs['parallax_over_error_th']
        else:
            parallax_over_error_th = 2
        savename='psfdata_synth_gaia_%s_%d_%d_p.pkl'%(fieldnamedict[ifield],m_min, m_max)
        profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='parallax_over_error',
                              m_min=m_min, m_max=m_max, filt_order=filt_order,
                               savename=savename, 
                               parallax_over_error_th=parallax_over_error_th)
    
        savename='psfdata_synth_gaia_%s_%d_%d_a.pkl'%(fieldnamedict[ifield],m_min, m_max)
        profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='radec_err',
                              m_min=m_min, m_max=m_max, filt_order=filt_order,
                               savename=savename)

    elif target_filter == 'astrometric_gof_al':
        savename='psfdata_synth_gaia_%s_%d_%d_g.pkl'%(fieldnamedict[ifield],m_min, m_max)
        profdatg = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_gof_al',
                              m_min=m_min, m_max=m_max, filt_order=filt_order,
                               savename=savename)
    elif target_filter == 'astrometric_excess_noise':    
        savename='psfdata_synth_gaia_%s_%d_%d_e.pkl'%(fieldnamedict[ifield],m_min, m_max)
        profdate = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_excess_noise',
                              m_min=m_min, m_max=m_max, filt_order=filt_order,
                               savename=savename)
    elif target_filter == 'parallax_over_error':    
        if 'parallax_over_error_th' in kwargs.keys():
            parallax_over_error_th = kwargs['parallax_over_error_th']
        else:
            parallax_over_error_th = 2
        savename='psfdata_synth_gaia_%s_%d_%d_p%d.pkl'%(fieldnamedict[ifield],m_min, m_max, parallax_over_error_th)
        profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='parallax_over_error',
                              m_min=m_min, m_max=m_max, filt_order=filt_order,
                               savename=savename,
                                parallax_over_error_th=parallax_over_error_th)
    elif target_filter == 'radec_err':
        savename='psfdata_synth_gaia_%s_%d_%d_a.pkl'%(fieldnamedict[ifield],m_min, m_max)
        profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='radec_err',
                              m_min=m_min, m_max=m_max, filt_order=filt_order,
                               savename=savename)


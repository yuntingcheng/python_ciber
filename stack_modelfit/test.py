# from reduction import *
# from psfsynth import *

# def run_gaia_test_stack(m_min, m_max, target_filter=None, **kwargs):
#     data_maps = {1: image_reduction(1), 2: image_reduction(2)}

#     inst = 1
#     filt_order = filt_order_dict[inst]
#     ifield = 6
    
#     if target_filter is None:
#         savename='psfdata_synth_gaia_%s_%d_%d_g.pkl'%(fieldnamedict[ifield],m_min, m_max)
#         profdatg = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_gof_al',
#                               m_min=m_min, m_max=m_max, filt_order=filt_order,
#                                savename=savename)

#         savename='psfdata_synth_gaia_%s_%d_%d_e.pkl'%(fieldnamedict[ifield],m_min, m_max)
#         profdate = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_excess_noise',
#                               m_min=m_min, m_max=m_max, filt_order=filt_order,
#                                savename=savename)

#         if 'parallax_over_error_th' in kwargs.keys():
#             parallax_over_error_th = kwargs['parallax_over_error_th']
#         else:
#             parallax_over_error_th = 2
#         savename='psfdata_synth_gaia_%s_%d_%d_p.pkl'%(fieldnamedict[ifield],m_min, m_max)
#         profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='parallax_over_error',
#                               m_min=m_min, m_max=m_max, filt_order=filt_order,
#                                savename=savename, 
#                                parallax_over_error_th=parallax_over_error_th)
    
#         savename='psfdata_synth_gaia_%s_%d_%d_a.pkl'%(fieldnamedict[ifield],m_min, m_max)
#         profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='radec_err',
#                               m_min=m_min, m_max=m_max, filt_order=filt_order,
#                                savename=savename)

#     elif target_filter == 'astrometric_gof_al':
#         savename='psfdata_synth_gaia_%s_%d_%d_g.pkl'%(fieldnamedict[ifield],m_min, m_max)
#         profdatg = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_gof_al',
#                               m_min=m_min, m_max=m_max, filt_order=filt_order,
#                                savename=savename)
#     elif target_filter == 'astrometric_excess_noise':    
#         savename='psfdata_synth_gaia_%s_%d_%d_e.pkl'%(fieldnamedict[ifield],m_min, m_max)
#         profdate = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='astrometric_excess_noise',
#                               m_min=m_min, m_max=m_max, filt_order=filt_order,
#                                savename=savename)
#     elif target_filter == 'parallax_over_error':    
#         if 'parallax_over_error_th' in kwargs.keys():
#             parallax_over_error_th = kwargs['parallax_over_error_th']
#         else:
#             parallax_over_error_th = 2
#         savename='psfdata_synth_gaia_%s_%d_%d_p%d.pkl'%(fieldnamedict[ifield],m_min, m_max, parallax_over_error_th)
#         profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='parallax_over_error',
#                               m_min=m_min, m_max=m_max, filt_order=filt_order,
#                                savename=savename,
#                                 parallax_over_error_th=parallax_over_error_th)
#     elif target_filter == 'radec_err':
#         savename='psfdata_synth_gaia_%s_%d_%d_a.pkl'%(fieldnamedict[ifield],m_min, m_max)
#         profdatp = stack_gaia(inst, ifield, data_maps=data_maps, target_filter='radec_err',
#                               m_min=m_min, m_max=m_max, filt_order=filt_order,
#                                savename=savename)

from srcmap import *
from mask import *
from stack import *
from reduction import *

def run_test_stack(inst, ifield):

    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    savename = './psf_stack_test'
    cbmap, strmask, mask_inst = load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                       (inst,ifield,'strmask'), 
                                       (inst,ifield,'mask_inst')])
    sigma_n = np.std(cbmap[strmask*mask_inst==1])
    make_srcmap_class = make_srcmap(inst, ifield=ifield)

    profdict = {}


    Nsrcs = np.array([1,3,5,10,20,50,100,200,300])
    Nsrc_tot = np.max(Nsrcs)
    xls = np.random.uniform(-0.5,1023.5, Nsrc_tot)
    yls = np.random.uniform(511.5, 512.5, Nsrc_tot)
    m_min_arr = np.array([13,15,17,19])
    m_max_arr = m_min_arr + 1
    start_time = time.time()
    for im, (m_min, m_max) in enumerate(zip(m_min_arr, m_max_arr)):
        profdict[im] = {}
        profdict[im]['m_min'] = m_min
        profdict[im]['m_max'] = m_max
        profdict[im]['prof'] = {}
        profdict[im]['profn'] = {}
        profdict[im]['profcb'] = {}
        stack_class = stacking_mock(inst, m_min, m_max, ifield=ifield)

        mags = np.random.uniform(m_min, m_max, Nsrc_tot)
        stack, stackn, stackcb = 0., 0., 0.
        for i,(m,x,y) in enumerate(zip(mags,xls,yls)):
            print('stack %d < m < %d,  %d / %d, t = %2.f min'\
                  %(m_min, m_max, i, Nsrc_tot,(time.time()-start_time)/60))
            make_srcmap_class.ms = np.array([m])
            make_srcmap_class.ms_inband = np.array([m])
            make_srcmap_class.xls = np.array([x])
            make_srcmap_class.yls = np.array([y])
            stack_class.xls = np.array([x])
            stack_class.yls = np.array([y])

            srcmap = make_srcmap_class.run_srcmap(ptsrc=True)

            _, stamp, _, _ = stack_class.run_stacking\
            (srcmap, np.ones_like(srcmap), np.zeros_like(srcmap), 
             mask_inst=mask_inst*strmask ,return_all=True, unmask=False)
            stack += stamp

            nmap = np.random.normal(scale=sigma_n,size=srcmap.shape)
            _, stamp, _, _ = stack_class.run_stacking\
            (srcmap+nmap, np.ones_like(srcmap), np.zeros_like(srcmap),
             return_all=True, unmask=False, mask_inst=mask_inst*strmask)
            stackn += stamp

            _, stamp, _, _ = stack_class.run_stacking\
            (srcmap+cbmap, np.ones_like(srcmap), np.zeros_like(srcmap),
             return_all=True, unmask=False, mask_inst=mask_inst*strmask)
            stackcb += stamp

            if i+1 in Nsrcs:
                profdict[im]['prof'][i+1] = radial_prof(stack/(i+1), stack_class.dx, stack_class.dx)
                profdict[im]['profn'][i+1] = radial_prof(stackn/(i+1), stack_class.dx, stack_class.dx)
                profdict[im]['profcb'][i+1] = radial_prof(stackcb/(i+1), stack_class.dx, stack_class.dx)

                with open(savename, "wb") as f:
                    pickle.dump(profdict, f)
    

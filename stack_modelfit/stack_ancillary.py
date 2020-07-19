from mask import *
from reduction import *
from clusters import *
import pandas as pd
from scipy.spatial import cKDTree
import time

def ps_src_select(inst, ifield, m_min, m_max, mask_insts, Nsub=64, sample_type='jack_random',
                  gaia_match=True, Nsrc_use=None, mask_clus=True, **kwargs):

    catdir = mypaths['PScatdat']
    df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')
    idx = np.arange(len(df))
    x1_arr, y1_arr = np.array(df['y1']), np.array(df['x1'])
    x2_arr, y2_arr = np.array(df['y2']), np.array(df['x2'])

    cls_arr = np.array(df['sdssClass'])
    cls_arr[cls_arr==3] = 1
    cls_arr[cls_arr==6] = -1

    photz_arr = np.array(df['Photz'])

    m_arr = np.array(df['I_comb'])
    if inst==1:
        m0_arr = np.array(df['I_comb'])
    else:
        m0_arr = np.array(df['H_comb'])

    sp = np.where((x1_arr>-0.5) & (x1_arr<1023.5) & (x2_arr>-0.5) & (x2_arr<1023.5) & \
                 (y1_arr>-0.5) & (y1_arr<1023.5) & (y2_arr>-0.5) & (y2_arr<1023.5))[0]

    x1_arr, y1_arr, x2_arr, y2_arr = x1_arr[sp], y1_arr[sp], x2_arr[sp], y2_arr[sp]
    m_arr, m0_arr, cls_arr, photz_arr = m_arr[sp], m0_arr[sp], cls_arr[sp], photz_arr[sp]
    idx = idx[sp]
    
    # count the center pix map
    centnum_map1, _, _ = np.histogram2d(x1_arr, y1_arr, np.arange(-0.5,1024.5,1))
    centnum_map2, _, _ = np.histogram2d(x2_arr, y2_arr, np.arange(-0.5,1024.5,1))

    spg = np.where((m_arr<=m_max) & (m_arr>m_min) & (cls_arr==1) & (photz_arr>=0))[0]
    sps = np.where((m_arr<=m_max) & (m_arr>m_min) & (cls_arr==-1))[0]
    sp = np.append(sps,spg)
    x1_arr, y1_arr, x2_arr, y2_arr = x1_arr[sp], y1_arr[sp], x2_arr[sp], y2_arr[sp]
    m_arr, m0_arr, z_arr = m_arr[sp], m0_arr[sp], photz_arr[sp]
    idx = idx[sp]
    cls_arr = np.ones(len(sp))
    cls_arr[:len(sps)] = -1

    # select sources not coexist with others in the same pixel 
    mask_inst1, mask_inst2 = mask_insts
    subm_arr, subm0_arr, subx1_arr, subx2_arr, suby1_arr, suby2_arr, \
    subz_arr, subcls_arr = [], [], [], [], [], [], [], []
    subidx_arr = []
    for i, (x1, y1, x2, y2,idxi) in enumerate(zip(x1_arr, y1_arr, x2_arr, y2_arr,idx)):
        if centnum_map1[int(np.round(x1)), int(np.round(y1))]==1 and \
        centnum_map2[int(np.round(x2)), int(np.round(y2))]==1 and \
        mask_inst1[int(np.round(x1)), int(np.round(y1))]==1 and \
        mask_inst2[int(np.round(x2)), int(np.round(y2))]==1:
            subm_arr.append(m_arr[i])
            subm0_arr.append(m0_arr[i])
            subz_arr.append(z_arr[i])
            subcls_arr.append(cls_arr[i])
            subx1_arr.append(x1)
            suby1_arr.append(y1)
            subx2_arr.append(x2)
            suby2_arr.append(y2)
            subidx_arr.append(idxi)
    subm_arr, subm0_arr, subz_arr, subcls_arr = \
    np.array(subm_arr), np.array(subm0_arr), np.array(subz_arr), np.array(subcls_arr) 
    subx1_arr, suby1_arr, subx2_arr, suby2_arr = \
    np.array(subx1_arr), np.array(suby1_arr), np.array(subx2_arr), np.array(suby2_arr)
    subidx_arr = np.array(subidx_arr)
    
    randidx = np.arange(len(subm_arr))
    np.random.shuffle(randidx)
    x1_arr, y1_arr = subx1_arr[randidx], suby1_arr[randidx]
    x2_arr, y2_arr = subx2_arr[randidx], suby2_arr[randidx]
    z_arr, m_arr, m0_arr, cls_arr =\
    subz_arr[randidx], subm_arr[randidx], subm0_arr[randidx], subcls_arr[randidx]
    idx_arr = subidx_arr[randidx]
    
    # mask clusters
    if mask_clus:
        maskmh = clusters(1, ifield, lnMhrange=(14, np.inf)).cluster_mask()
        maskz = clusters(1, ifield, zrange=(0, 0.15)).cluster_mask()  
        clus_mask1 = maskz * maskmh
        
        maskmh = clusters(2, ifield, lnMhrange=(14, np.inf)).cluster_mask()
        maskz = clusters(2, ifield, zrange=(0, 0.15)).cluster_mask()  
        clus_mask2 = maskz * maskmh
        
        subm_arr, subm0_arr, subx1_arr, suby1_arr, subx2_arr, suby2_arr, subz_arr, subcls_arr =\
        [], [], [], [], [], [], [], []
        subidx_arr = []
        for i, (x1,y1,x2,y2) in enumerate(zip(x1_arr,y1_arr,x2_arr,y2_arr)):
            if (clus_mask1[int(np.round(x1)), int(np.round(y1))]==1) or \
            (clus_mask2[int(np.round(x2)), int(np.round(y2))]==1):
                subm_arr.append(m_arr[i])
                subm0_arr.append(m0_arr[i])
                subz_arr.append(z_arr[i])
                subcls_arr.append(cls_arr[i])
                subx1_arr.append(x1)
                suby1_arr.append(y1)
                subx2_arr.append(x2)
                suby2_arr.append(y2)
                subidx_arr.append(idx_arr[i])
        x1_arr, y1_arr, z_arr = np.array(subx1_arr), np.array(suby1_arr), np.array(subz_arr)        
        x2_arr, y2_arr = np.array(subx2_arr), np.array(suby2_arr)
        m_arr, m0_arr, cls_arr = np.array(subm_arr), np.array(subm0_arr), np.array(subcls_arr) 
        idx_arr = np.array(subidx_arr)
    
    if inst == 1:
        x_arr, y_arr = x1_arr, y1_arr
    else:
        x_arr, y_arr = x2_arr, y2_arr
        
    xg_arr, yg_arr, mg_arr, mg0_arr =\
    x_arr[cls_arr==1], y_arr[cls_arr==1], m_arr[cls_arr==1], m0_arr[cls_arr==1]
    zg_arr = z_arr[cls_arr==1]
    idxg_arr = idx_arr[cls_arr==1]
    
    if mask_clus:
        sp = np.where(zg_arr>0.15)[0]
        xg_arr, yg_arr, zg_arr, mg_arr, mg0_arr = \
        xg_arr[sp], yg_arr[sp], zg_arr[sp], mg_arr[sp], mg0_arr[sp]
        idxg_arr = idxg_arr[sp]
    
    xs_arr, ys_arr, ms_arr, ms0_arr =\
    x_arr[cls_arr==-1], y_arr[cls_arr==-1], m_arr[cls_arr==-1], m0_arr[cls_arr==-1]
    idxs_arr = idx_arr[cls_arr==-1]
    
    if gaia_match:
        dfg = pd.read_csv(mypaths['GAIAcatdat'] + fieldnamedict[ifield] + '.csv')
        dfg = dfg[dfg['parallax']==dfg['parallax']]
        catalogg = (dfg[['ra','dec']].values * np.pi/180).tolist()
        psg = [[item[0], item[1]] for item in catalogg]
        dfp = df.iloc[idxg_arr]
        catalogp = (df[['ra','dec']].iloc[idxg_arr].values * np.pi/180).tolist()
        psp = [[item[0], item[1]] for item in catalogp]
        kdt = cKDTree(psg)
        obj = kdt.query_ball_point(psp, (700 * u.mas).to(u.rad).value)
        Nmatch = np.array([len(obj_i) for obj_i in obj])
        xg_arr, yg_arr, mg_arr, mg0_arr =\
        xg_arr[Nmatch==0], yg_arr[Nmatch==0], mg_arr[Nmatch==0], mg0_arr[Nmatch==0]
        zg_arr = zg_arr[Nmatch==0]
        idxg_arr = idxg_arr[Nmatch==0]

        dfg = pd.read_csv(mypaths['GAIAcatdat'] + fieldnamedict[ifield] + '.csv')
        dfg = dfg[(dfg['parallax']==dfg['parallax']) \
        & (dfg['astrometric_excess_noise']==0)]
        catalogg = (dfg[['ra','dec']].values * np.pi/180).tolist()
        psg = [[item[0], item[1]] for item in catalogg]
        dfp = df.iloc[idxs_arr]
        catalogp = (df[['ra','dec']].iloc[idxs_arr].values * np.pi/180).tolist()
        psp = [[item[0], item[1]] for item in catalogp]
        kdt = cKDTree(psg)
        obj = kdt.query_ball_point(psp, (1. * u.arcsec).to(u.rad).value)
        Nmatch = np.array([len(obj_i) for obj_i in obj])
        xs_arr, ys_arr, ms_arr, ms0_arr =\
        xs_arr[Nmatch>0], ys_arr[Nmatch>0], ms_arr[Nmatch>0], ms0_arr[Nmatch>0]
        idxs_arr = idxs_arr[Nmatch>0]
        
    srcdat = {}
    srcdat['inst']= inst
    srcdat['ifield'] = ifield
    srcdat['field'] = fieldnamedict[ifield]
    srcdat['sample_type'] = sample_type
    srcdat['m_min'], srcdat['m_max'] = m_min, m_max
    srcdat['Ng'], srcdat['Ns'] = len(xg_arr), len(xs_arr)

    if Nsrc_use is not None:
        if srcdat['Ng'] > Nsrc_use:
            sp = np.random.choice(srcdat['Ng'], Nsrc_use, replace=False)
            xg_arr, yg_arr = xg_arr[sp], yg_arr[sp]
            mg_arr, mg0_arr, zg_arr = mg_arr[sp], mg0_arr[sp], zg_arr[sp]
            idxg_arr = idxg_arr[sp]
        if srcdat['Ns'] > Nsrc_use:
            sp = np.random.choice(srcdat['Ns'], Nsrc_use, replace=False)
            xs_arr, ys_arr = xs_arr[sp], ys_arr[sp]
            ms_arr, ms0_arr = ms_arr[sp], ms0_arr[sp]
            idxs_arr = idxs_arr[sp]

    if sample_type == 'all':
        srcdat['xg_arr'], srcdat['yg_arr'] = xg_arr, yg_arr
        srcdat['mg_arr'], srcdat['zg_arr'] = mg_arr, zg_arr
        srcdat['mg0_arr'] = mg0_arr
        srcdat['xs_arr'], srcdat['ys_arr'] = xs_arr, ys_arr
        srcdat['ms_arr'] = ms_arr
        srcdat['ms0_arr'] = ms0_arr
        srcdat['idxg_arr'] = idxg_arr
        srcdat['idxs_arr'] = idxs_arr
        
    elif sample_type == 'jack_random':
        srcdat['Nsub'] = Nsub
        srcdat['sub'] = {}
        for i in range(Nsub):
            srcdat['sub'][i] = {}
            spg = np.arange(i,len(xg_arr),Nsub)
            srcdat['sub'][i]['xg_arr'], srcdat['sub'][i]['yg_arr'] = xg_arr[spg], yg_arr[spg]
            srcdat['sub'][i]['mg_arr'], srcdat['sub'][i]['zg_arr'] = mg_arr[spg], zg_arr[spg]
            srcdat['sub'][i]['mg0_arr'] = mg0_arr[spg]
            srcdat['sub'][i]['idxg_arr'] = idxg_arr[spg]
            sps = np.arange(i,len(xs_arr),Nsub)
            srcdat['sub'][i]['xs_arr'], srcdat['sub'][i]['ys_arr'] = xs_arr[sps], ys_arr[sps]
            srcdat['sub'][i]['ms_arr'] = ms_arr[sps]
            srcdat['sub'][i]['ms0_arr'] = ms0_arr[sps]
            srcdat['sub'][i]['idxs_arr'] = idxs_arr[sps]
            srcdat['sub'][i]['Ng'], srcdat['sub'][i]['Ns'] = len(spg), len(sps)
    
    elif sample_type == 'jack_region':
        srcdat['Nsub'] = Nsub
        srcdat['sub'] = {}
        Nsides = int(np.sqrt(Nsub))
        axlims = np.linspace(-0.5, 1023.5, Nsides+1)
        ymins, xmins = np.meshgrid(axlims[:-1], axlims[:-1])
        ymaxs, xmaxs = np.meshgrid(axlims[1:], axlims[1:])
        for i in range(Nsub):
            srcdat['sub'][i] = {}
            ymin, xmin = ymins.flatten()[i], xmins.flatten()[i]
            ymax, xmax = ymaxs.flatten()[i], xmaxs.flatten()[i]
            spg = np.where((xg_arr>=xmin) & (xg_arr<xmax) \
                           & (yg_arr>=ymin) & (yg_arr<ymax))[0]
            srcdat['sub'][i]['xg_arr'], srcdat['sub'][i]['yg_arr'] = xg_arr[spg], yg_arr[spg]
            srcdat['sub'][i]['mg_arr'], srcdat['sub'][i]['zg_arr'] = mg_arr[spg], zg_arr[spg]
            srcdat['sub'][i]['mg0_arr'] = mg0_arr[spg]
            srcdat['sub'][i]['idxg_arr'] = idxg_arr[spg]
            sps = np.where((xs_arr>=xmin) & (xs_arr<xmax) \
                           & (ys_arr>=ymin) & (ys_arr<ymax))[0]
            srcdat['sub'][i]['xs_arr'], srcdat['sub'][i]['ys_arr'] = xs_arr[sps], ys_arr[sps]
            srcdat['sub'][i]['ms_arr'] = ms_arr[sps]
            srcdat['sub'][i]['ms0_arr'] = ms0_arr[sps]
            srcdat['sub'][i]['idxs_arr'] = idxs_arr[sps]
            srcdat['sub'][i]['Ng'], srcdat['sub'][i]['Ns'] = len(spg), len(sps)
            
    return srcdat

def tm_src_select(inst, ifield, m_min, m_max, mask_insts, band_select='I',
                  Nsub=64, sample_type='jack_random', Nsrc_use=None):
    catdir = mypaths['2Mcatdat']
    df = pd.read_csv(catdir + fieldnamedict[ifield] + '.csv')

    x1_arr, y1_arr = np.array(df['y1']), np.array(df['x1'])
    x2_arr, y2_arr = np.array(df['y2']), np.array(df['x2'])

    m_arr = np.array(df[band_select])
    if inst==1:
        m0_arr = np.array(df['I'])
    else:
        m0_arr = np.array(df['H'])

    sp = np.where((x1_arr>-0.5) & (x1_arr<1023.5) & (x2_arr>-0.5) & (x2_arr<1023.5) & \
                 (y1_arr>-0.5) & (y1_arr<1023.5) & (y2_arr>-0.5) & (y2_arr<1023.5))[0]

    x1_arr, y1_arr, x2_arr, y2_arr = x1_arr[sp], y1_arr[sp], x2_arr[sp], y2_arr[sp]
    m_arr, m0_arr = m_arr[sp], m0_arr[sp]

    # count the center pix map
    centnum_map1, _, _ = np.histogram2d(x1_arr, y1_arr, np.arange(-0.5,1024.5,1))
    centnum_map2, _, _ = np.histogram2d(x2_arr, y2_arr, np.arange(-0.5,1024.5,1))

    sp = np.where((m_arr<=m_max) & (m_arr>m_min))[0]
    x1_arr, y1_arr, x2_arr, y2_arr = x1_arr[sp], y1_arr[sp], x2_arr[sp], y2_arr[sp]
    m_arr, m0_arr, z_arr = m_arr[sp], m0_arr[sp], photz_arr[sp]

    # select sources not coexist with others in the same pixel 
    mask_inst1, mask_inst2 = mask_insts
    subm_arr, subm0_arr, subx1_arr, subx2_arr, suby1_arr, suby2_arr = [], [], [], [], [], []
    for i, (x1, y1, x2, y2) in enumerate(zip(x1_arr, y1_arr, x2_arr, y2_arr)):
        if centnum_map1[int(np.round(x1)), int(np.round(y1))]==1 and \
        centnum_map2[int(np.round(x2)), int(np.round(y2))]==1 and \
        mask_inst1[int(np.round(x1)), int(np.round(y1))]==1 and \
        mask_inst2[int(np.round(x2)), int(np.round(y2))]==1:
            subm_arr.append(m_arr[i])
            subm0_arr.append(m0_arr[i])
            subx1_arr.append(x1)
            suby1_arr.append(y1)
            subx2_arr.append(x2)
            suby2_arr.append(y2)
    subm_arr, subm0_arr = np.array(subm_arr), np.array(subm0_arr)
    subx1_arr, suby1_arr, subx2_arr, suby2_arr = \
    np.array(subx1_arr), np.array(suby1_arr), np.array(subx2_arr), np.array(suby2_arr)

    randidx = np.arange(len(subm_arr))
    np.random.shuffle(randidx)
    if inst==1:
        x_arr, y_arr = subx1_arr[randidx], suby1_arr[randidx]
    else:
        x_arr, y_arr = subx2_arr[randidx], suby2_arr[randidx]

    m_arr, m0_arr = subm_arr[randidx], subm0_arr[randidx]
    xs_arr, ys_arr, ms_arr, ms0_arr = x_arr, y_arr, m_arr, m0_arr

    srcdat = {}
    srcdat['inst']= inst
    srcdat['ifield'] = ifield
    srcdat['field'] = fieldnamedict[ifield]
    srcdat['sample_type'] = sample_type
    srcdat['m_min'], srcdat['m_max'] = m_min, m_max
    srcdat['Ns'] = len(xs_arr)

    if Nsub is None:
        Nsub = len(xs_arr)

    if Nsrc_use is not None:
        if srcdat['Ns'] > Nsrc_use:
            sp = np.random.choice(srcdat['Ns'], Nsrc_use, replace=False)
            xs_arr, ys_arr = xs_arr[sp], ys_arr[sp]
            ms_arr, ms0_arr = ms_arr[sp], ms0_arr[sp]

    if sample_type == 'all':
        srcdat['xs_arr'], srcdat['ys_arr'] = xs_arr, ys_arr
        srcdat['ms_arr'] = ms_arr
        srcdat['ms0_arr'] = ms0_arr
        
    elif sample_type == 'jack_random':
        srcdat['Nsub'] = Nsub
        srcdat['sub'] = {}
        for i in range(Nsub):
            srcdat['sub'][i] = {}
            sps = np.arange(i,len(xs_arr), Nsub)
            srcdat['sub'][i]['xs_arr'], srcdat['sub'][i]['ys_arr'] = xs_arr[sps], ys_arr[sps]
            srcdat['sub'][i]['ms_arr'] = ms_arr[sps]
            srcdat['sub'][i]['ms0_arr'] = ms0_arr[sps]
            srcdat['sub'][i]['Ns'] = len(sps)
    
    elif sample_type == 'jack_region':
        srcdat['Nsub'] = Nsub
        srcdat['sub'] = {}
        Nsides = int(np.sqrt(Nsub))
        axlims = np.linspace(-0.5, 1023.5, Nsides+1)
        ymins, xmins = np.meshgrid(axlims[:-1], axlims[:-1])
        ymaxs, xmaxs = np.meshgrid(axlims[1:], axlims[1:])
        for i in range(Nsub):
            srcdat['sub'][i] = {}
            ymin, xmin = ymins.flatten()[i], xmins.flatten()[i]
            ymax, xmax = ymaxs.flatten()[i], xmaxs.flatten()[i]
            sps = np.where((xs_arr>=xmin) & (xs_arr<xmax) \
                           & (ys_arr>=ymin) & (ys_arr<ymax))[0]
            srcdat['sub'][i]['xs_arr'], srcdat['sub'][i]['ys_arr'] = xs_arr[sps], ys_arr[sps]
            srcdat['sub'][i]['ms_arr'] = ms_arr[sps]
            srcdat['sub'][i]['ms0_arr'] = ms0_arr[sps]
            srcdat['sub'][i]['Ns'] = len(sps)
            
    return srcdat

def run_nonuniform_BG(inst, ifield):
    for im in range(4):
        m_min, m_max = magbindict['m_min'][im],magbindict['m_max'][im]
        stack = stacking(inst, ifield, m_min, m_max, 
            load_from_file=True,run_nonuniform_BG=True)
        

def stack_bigpix(inst, ifield, m_min, m_max, srctype='g', dx=120,
                 sample_type='jack_random', filt_order=None, verbose=False):

    stackdat = {}
    data_maps = {1: image_reduction(1), 2: image_reduction(2)}
    cbmap, strmask, strnum, mask_inst1, mask_inst2 = \
    load_processed_images(data_maps, return_names=[(inst,ifield,'cbmap'), 
                                       (inst,ifield,'strmask'), 
                                       (inst,ifield,'strnum'),
                                       (1,ifield,'mask_inst'),
                                       (2,ifield,'mask_inst')])
    if inst==1:
        mask_inst = mask_inst1
    else:
        mask_inst = mask_inst2
            
    cbmap = image_poly_filter(cbmap, strmask*mask_inst, degree=filt_order)
    srcdat = ps_src_select(inst, ifield, m_min, m_max, 
                           [mask_inst1, mask_inst2], sample_type=sample_type)

    if inst==1:
        mask_inst = mask_inst1
    else:
        mask_inst = mask_inst2

    dx = 1200
    profile = radial_prof(np.ones([2*dx+1,2*dx+1]), dx, dx)
    rbinedges, rbins = profile['rbinedges'], profile['rbins']
    stackdat['rbins'] = rbins*0.7
    stackdat['rbinedges'] = rbinedges*0.7
    rbins /= 10 # bigpix
    rbinedges /=10 # bigpix

    dx = 120
    radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx)
    cbmapi = cbmap*strmask*mask_inst
    maski = strmask*mask_inst
    
    Nsides = int(np.sqrt(srcdat['Nsub']))
    axlims = np.linspace(-0.5, 1023.5, Nsides+1)
    ymins, xmins = np.meshgrid(axlims[:-1], axlims[:-1])
    ymaxs, xmaxs = np.meshgrid(axlims[1:], axlims[1:])

    cbmapstack = np.zeros([2*dx+1,2*dx+1])
    maskstack = np.zeros([2*dx+1,2*dx+1])
    start_time = time.time()
    for isub in range(srcdat['Nsub']):
        
        if srctype == 'g':
            Nsrc = srcdat['sub'][isub]['Ng']
            x_arr, y_arr = srcdat['sub'][isub]['xg_arr'], srcdat['sub'][isub]['yg_arr']
        elif srctype == 's':
            Nsrc = srcdat['sub'][isub]['Ns']
            x_arr, y_arr = srcdat['sub'][isub]['xs_arr'], srcdat['sub'][isub]['ys_arr']
        elif srctype == 'bg':
            Nsrc = srcdat['sub'][isub]['Ng']
            if sample_type=='jack_random':
                x_arr = np.random.randint(-0.5,1023.5,Nsrc)
                y_arr = np.random.randint(-0.5,1023.5,Nsrc)
            elif sample_type=='jack_region':
                ymin, xmin = ymins.flatten()[isub], xmins.flatten()[isub]
                ymax, xmax = ymaxs.flatten()[isub], xmaxs.flatten()[isub]
                x_arr = np.random.randint(xmin,xmax,Nsrc)
                y_arr = np.random.randint(ymin,ymax,Nsrc)

        stackdat[isub] = {}
        if verbose:
            print('stacking %s %d < m < %d, #%d, %d src, t = %.2f min'\
              %(fieldnamedict[ifield], m_min, m_max, isub, 
                Nsrc, (time.time()-start_time)/60))

        cbmapstacki = np.zeros([2*dx+1,2*dx+1])
        maskstacki = np.zeros([2*dx+1,2*dx+1])
        for i in range(Nsrc):
            xi, yi = int(round(x_arr[i])), int(round(y_arr[i]))
            radmap = make_radius_map(cbmap, xi, yi) # large pix units

            # zero padding
            mcb = np.pad(cbmapi, ((dx,dx),(dx,dx)), 'constant')
            k = np.pad(maski, ((dx,dx),(dx,dx)), 'constant')
            xi += dx
            yi += dx

            # cut stamp
            cbmapstamp = mcb[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]
            maskstamp = k[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]

            cbmapstacki += cbmapstamp
            maskstacki += maskstamp

        ### end source for loop ###
        cbmapstack += cbmapstacki
        maskstack += maskstacki

        Nbins = len(rbins)
        profcb_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
        for ibin in range(Nbins):
            spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                           (radmapstamp<rbinedges[ibin+1]))
            profcb_arr[ibin] += np.sum(cbmapstacki[spi])
            hit_arr[ibin] += np.sum(maskstacki[spi])
        profcbnorm = np.zeros_like(profcb_arr)
        profcbnorm[hit_arr!=0] = profcb_arr[hit_arr!=0]/hit_arr[hit_arr!=0]
        stackdat[isub]['profcb'] = profcbnorm
        stackdat[isub]['profhit'] = hit_arr

    profcb_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
    for ibin in range(Nbins):
        spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                       (radmapstamp<rbinedges[ibin+1]))
        profcb_arr[ibin] += np.sum(cbmapstack[spi])
        hit_arr[ibin] += np.sum(maskstack[spi])
    profcb_norm = np.zeros_like(profcb_arr)
    profcb_norm[hit_arr!=0] = profcb_arr[hit_arr!=0]/hit_arr[hit_arr!=0]
    stackdat['profcb'] = profcb_norm
    stackdat['profhit'] = hit_arr

    data_cb = np.zeros([srcdat['Nsub'], Nbins])
    stackdat['jack'] = {}
    for isub in range(srcdat['Nsub']):
        stackdat['jack'][isub] = {}
        profcb = stackdat['profcb']*stackdat['profhit'] - \
        stackdat[isub]['profcb']*stackdat[isub]['profhit']
        profhit = stackdat['profhit'] - stackdat[isub]['profhit']
        stackdat['jack'][isub]['profcb'] = profcb/profhit
        stackdat['jack'][isub]['profhit'] = profhit
        data_cb[isub,:] = profcb/profhit

    stackdat['profcb_err'] = np.sqrt(np.nanvar(data_cb, axis=0)*(srcdat['Nsub']-1))

    return stackdat


class stacking_mock:
    def __init__(self, inst, m_min=16, m_max=17, srctype='g', 
                 catname = 'PanSTARRS', ifield = 8, pixsize=7):
        self.inst = inst
        self.m_min = m_min
        self.m_max = m_max
        self.pixsize = pixsize
        
        if catname=='PanSTARRS':
            self.catname = catname
            self.catdir = mypaths['PScatdat']
            if ifield in [4,5,6,7,8]:
                self.ifield = ifield
                self.field = fieldnamedict[ifield]
            else:
                raise Exception('ifield invalid (must be int between 4-8)')
        elif catname=='HSC':
            self.catname = catname
            self.catdir = mypaths['HSCcatdat']
            if ifield in range(12):
                self.ifield = ifield
                self.field = 'W05_%d'%ifield
            else:
                raise Exception('ifield invalid (must be int between 0-11)')
        else:
            raise Exception('catname invalid (enter PanSTARRS or HSC)')
        
        if srctype in [None, 's', 'g', 'u']:
            self.srctype = srctype
        else:
            raise Exception('srctype invalid (must be None, "s", "g", or "u")')
        
        self.Npix_cb = 1024
        self.Nsub = 10
        
        self.xls, self.yls, self.ms, self.xss, self.yss, self.ms_inband = self._load_catalog()
        
        self._get_mask_radius()
        
    def _load_catalog(self):
        Npix_cb = self.Npix_cb
        Nsub = self.Nsub
        
        df = pd.read_csv(self.catdir + self.field + '.csv')
        if self.catname == 'PanSTARRS':
            xls = df['y'+str(self.inst)].values
            yls = df['x'+str(self.inst)].values
            ms = df['I_comb'].values
            clss = df['sdssClass'].values
            if self.srctype == 's':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==6))[0]
            elif self.srctype == 'g':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==3))[0]
            elif self.srctype == 'u':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==-99))[0]
            elif self.srctype is None:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max))[0]            

        else:
            xls = df['x'].values - 2.5
            yls = df['y'].values - 2.5
            ms = df['Imag'].values
            clss = df['cls'].values
            if self.srctype == 's':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==-1))[0]
            elif self.srctype == 'g':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==1))[0]
            elif self.srctype == 'u':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==2))[0]
            elif self.srctype is None:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max))[0]            
            
        xls = xls[sp]
        yls = yls[sp]
        clss = clss[sp]
        ms = ms[sp]
        ms_inband = ms.copy()
        if self.inst == 2:
            if self.catname == 'PanSTARRS':
                ms_inband = df['H_comb'].values[sp]
            else:
                ms_inband = df['Hmag'].values[sp]
            
        sp = np.where((xls > 0) & (yls > 0) \
                      & (xls < Npix_cb) & (yls < Npix_cb))[0]
        xls = xls[sp]
        yls = yls[sp]
        ms = ms[sp]
        ms_inband = ms_inband[sp]

        xss = np.round(xls * Nsub + (Nsub/2 - 0.5)).astype(np.int32)
        yss = np.round(yls * Nsub + (Nsub/2 - 0.5)).astype(np.int32)

        return xls, yls, ms, xss, yss, ms_inband
    
    def _get_mask_radius(self, mask_func = 'Ith_mask', ifield=8, Ith=1):
        if mask_func == 'MZ14_mask':
            rs =  MZ14_mask(self.inst, self.xls, self.yls, self.ms_inband,
                            return_radius=True)
        elif mask_func == 'Ith_mask':
            rs = get_mask_radius_th(ifield, self.ms, inst=1, Ith=Ith)
        
        self.rs = rs
        
    def _image_finegrid(self, image):
        Nsub = self.Nsub
        w, h  = np.shape(image)
        image_new = np.zeros([w*Nsub, h*Nsub])
        for i in range(Nsub):
            for j in range(Nsub):
                image_new[i::Nsub, j::Nsub] = image
        return image_new

    def run_stacking(self, mapin, mask, num, mask_inst=None,
                     dx=1200, return_profile = True, return_all=False,
                      update_mask=True,cliplim=None, verbose=True):
        
        self.dx = dx
        Nsub = self.Nsub
        
        self.xss = np.round(self.xls * Nsub + (Nsub/2 - 0.5)).astype(np.int32)
        self.yss = np.round(self.yls * Nsub + (Nsub/2 - 0.5)).astype(np.int32)
        if update_mask:
            self._get_mask_radius()
        
        mask_inst = np.ones_like(mapin) if mask_inst is None else mask_inst
        
        if cliplim is not None:
            profile = radial_prof(np.ones([2*dx+1,2*dx+1]), dx, dx)
            rbinedges, rbins = profile['rbinedges'], profile['rbins']
            Nbins = len(rbins)
            rbinedges = rbinedges/0.7 # subpix unit

        mapstack = 0.
        maskstack = 0
        for i,(xl,yl,xs,ys,r) in enumerate(zip(self.xls,self.yls,self.xss,self.yss,self.rs)):
            if len(self.xls)>20:
                if verbose and i%(len(self.xls)//20)==0:
                    print('stacking %d / %d (%.1f %%)'\
                          %(i, len(self.xls), i/len(self.xls)*100))
            
            # unmask source
            radmap = make_radius_map(mapin, xl, yl)
            maski = mask*mask_inst
            m = mapin * maski
            sp1 = np.where((radmap < r / self.pixsize) & (num==1) & (mask_inst==1))
            m[sp1] = mapin[sp1]
            maski[sp1] = 1
            unmaskpix = np.zeros_like(mask)
            unmaskpix[sp1] = 1
            if len(sp1[0]) > 0 and cliplim is not None:
                for ibin in range(Nbins):
                    if cliplim['CBmax'][ibin] == np.inf:
                        continue
                    spi = np.where((unmaskpix==1) & \
                                   (radmap*10>=rbinedges[ibin]) & \
                                   (radmap*10 < rbinedges[ibin+1]) & \
                                   (mapin > cliplim['CBmax'][ibin]))
                    m[spi] = 0
                    maski[spi] = 0
                    spi = np.where((unmaskpix==1) & \
                                   (radmap*10>=rbinedges[ibin]) & \
                                   (radmap*10 < rbinedges[ibin+1]) & \
                                   (mapin < cliplim['CBmin'][ibin]))
                    m[spi] = 0
                    maski[spi] = 0

            m = self._image_finegrid(m)
            k = self._image_finegrid(maski)
            
            # zero padding
            m = np.pad(m, ((dx,dx),(dx,dx)), 'constant')
            k = np.pad(k, ((dx,dx),(dx,dx)), 'constant')
            xs += dx
            ys += dx
            
            # cut stamp
            mapstamp = m[xs - dx: xs + dx + 1, ys - dx: ys + dx + 1]
            maskstamp = k[xs - dx: xs + dx + 1, ys - dx: ys + dx + 1]
            
            mapstack += mapstamp
            maskstack += maskstamp
            
        stack = np.zeros_like(mapstack)
        sp = np.where(maskstack!=0)
        stack[sp] = mapstack[sp] / maskstack[sp]
        stack[maskstack==0] = 0
        
        if not return_profile and not return_all: 
            return stack, maskstack, mapstack
        
        stackdat = {}
        profile = radial_prof(np.ones([2*dx+1,2*dx+1]), dx, dx)
        rbinedges, rbins = profile['rbinedges'], profile['rbins']
        rsubbins, rsubbinedges = self._radial_binning(rbins, rbinedges)
        Nbins = len(rbins)
        Nsubbins = len(rsubbins)

        stackdat['rbins'] = rbins*0.7
        stackdat['rbinedges'] = rbinedges*0.7     
        stackdat['rsubbins'] = rsubbins*0.7
        stackdat['rsubbinedges'] = rsubbinedges*0.7

        radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx)
        prof_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
        for ibin in range(Nbins):
            spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                           (radmapstamp<rbinedges[ibin+1]))
            prof_arr[ibin] += np.sum(mapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        prof_norm = np.zeros_like(prof_arr)
        prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]
        stackdat['prof'] = prof_norm
        stackdat['profhit'] = hit_arr

        prof_arr, hit_arr = np.zeros(Nsubbins), np.zeros(Nsubbins)
        for ibin in range(Nsubbins):
            spi = np.where((radmapstamp>=rsubbinedges[ibin]) &\
                           (radmapstamp<rsubbinedges[ibin+1]))
            prof_arr[ibin] += np.sum(mapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        prof_norm = np.zeros_like(prof_arr)
        prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]
        stackdat['profsub'] = prof_norm
        stackdat['profhitsub'] = hit_arr

        if return_profile and not return_all: 
            return stackdat

        if return_all:
            return stackdat, stack, maskstack, mapstack

    def run_stacking_bigpix(self, mapin, mask, num, mask_inst=None,
                     dx=120, return_profile = True, verbose=True):
        
        self.dx = dx
                
        mask_inst = np.ones_like(mapin) if mask_inst is None else mask_inst
        maski = mask * mask_inst
        mapi = mapin * maski
        
        mapstack = 0.
        maskstack = 0
        for i,(xi,yi) in enumerate(zip(self.xls,self.yls)):
            xi, yi = int(round(xi)), int(round(yi))
            if len(self.xls)>20:
                if verbose and i%(len(self.xls)//20)==0:
                    print('stacking %d / %d (%.1f %%)'\
                          %(i, len(self.xls), i/len(self.xls)*100))
            
            radmap = make_radius_map(mapi, xi, yi) # large pix units
            
            # zero padding
            m = np.pad(mapi, ((dx,dx),(dx,dx)), 'constant')
            k = np.pad(maski, ((dx,dx),(dx,dx)), 'constant')
            xi += dx
            yi += dx

            # cut stamp
            mapstamp = m[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]
            maskstamp = k[xi - dx: xi + dx + 1, yi - dx: yi + dx + 1]

            mapstack += mapstamp
            maskstack += maskstamp
            
        stack = np.zeros_like(mapstack)
        sp = np.where(maskstack!=0)
        stack[sp] = mapstack[sp] / maskstack[sp]
        stack[maskstack==0] = 0
        
        if not return_profile: 
            return stack, maskstack, mapstack
        
        profile = radial_prof(np.ones([2*dx*10+1,2*dx*10+1]), dx*10, dx*10)
        rbinedges, rbins = profile['rbinedges'], profile['rbins']
        rsubbins, rsubbinedges = self._radial_binning(rbins, rbinedges)
        Nbins = len(rbins)
        Nsubbins = len(rsubbins)
        stackdat = {}
        stackdat['rbins'] = rbins*0.7
        stackdat['rbinedges'] = rbinedges*0.7     
        stackdat['rsubbins'] = rsubbins*0.7
        stackdat['rsubbinedges'] = rsubbinedges*0.7

        rbins /= 10 # bigpix
        rbinedges /=10 # bigpix
        radmapstamp =  make_radius_map(np.zeros((2*dx+1, 2*dx+1)), dx, dx)
        prof_arr, hit_arr = np.zeros(Nbins), np.zeros(Nbins)
        for ibin in range(Nbins):
            spi = np.where((radmapstamp>=rbinedges[ibin]) &\
                           (radmapstamp<rbinedges[ibin+1]))
            prof_arr[ibin] += np.sum(mapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        prof_norm = np.zeros_like(prof_arr)
        prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]       
        stackdat['prof'] = prof_norm
        stackdat['profhit'] = hit_arr

        rsubbins /= 10 # bigpix
        rsubbinedges /=10 # bigpix
        prof_arr, hit_arr = np.zeros(Nsubbins), np.zeros(Nsubbins)
        for ibin in range(Nsubbins):
            spi = np.where((radmapstamp>=rsubbinedges[ibin]) &\
                           (radmapstamp<rsubbinedges[ibin+1]))
            prof_arr[ibin] += np.sum(mapstack[spi])
            hit_arr[ibin] += np.sum(maskstack[spi])
        prof_norm = np.zeros_like(prof_arr)
        prof_norm[hit_arr!=0] = prof_arr[hit_arr!=0]/hit_arr[hit_arr!=0]       
        stackdat['profsub'] = prof_norm
        stackdat['profhitsub'] = hit_arr

        return stackdat

    def _radial_binning(self,rbins,rbinedges):
        rsubbinedges = np.concatenate((rbinedges[:1],rbinedges[6:20],rbinedges[-1:]))

        # calculate 
        rin = (2./3) * (rsubbinedges[1]**3 - rsubbinedges[0]**3)\
        / (rsubbinedges[1]**2 - rsubbinedges[0]**2)

        rout = (2./3) * (rsubbinedges[-1]**3 - rsubbinedges[-2]**3)\
        / (rsubbinedges[-1]**2 - rsubbinedges[-2]**2)

        rsubbins = np.concatenate(([rin],rbins[6:19],[rout]))

        return rsubbins, rsubbinedges

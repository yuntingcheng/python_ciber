from reduction import *
from psfsynth import *

def run_gaia_test_stack(m_min, m_max):
    data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    inst = 1
    ifield = 6
    df = pd.read_csv(mypaths['GAIAcatdat'] + fieldnamedict[ifield] + '_raw.csv')
    df = catalog_add_xy_from_radec(fieldnamedict[ifield], df)
    df = df.loc[df['parallax']==df['parallax']]
    df = df.loc[df['parallax'] > 1/5e3]
    df2 = df.loc[(df['parallax_over_error']>2)]
    df0 = df.loc[(df['astrometric_excess_noise']==0)]

    savename='psfdata_synth_gaia_%s_%d_%d_0.pkl'%(fieldnamedict[ifield],m_min, m_max)
    profdat0 = stack_gaia(inst, ifield, data_maps=data_maps, df=df0,
                          m_min=m_min, m_max=m_max, savename=savename)
    avename='psfdata_synth_gaia_%s_%d_%d_2.pkl'%(fieldnamedict[ifield],m_min, m_max)
    profdat2 = stack_gaia(inst, ifield, data_maps=data_maps, df=df2, 
                          m_min=m_min, m_max=m_max, savename=savename)

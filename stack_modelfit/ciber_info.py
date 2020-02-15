from ciber_paths import *
from utils_plotting import *

fieldnamedict = {4:'elat10',
                 5:'elat30',
                 6:'BootesB',
                 7:'BootesA',
                 8:'SWIRE'}

magbindict = {'m_min':[16,17,18,19],
              'm_max':[17,18,19,20],
              'zmean': [0.13, 0.20, 0.27, 0.43],
              'zmed': [0.11, 0.19, 0.26, 0.41]
             }

cal_factor_dict = {'framerate': 1/(1.6e-6 * 512 * 513 * 68 / 16), # 0.5599 frame / s
                   'apf2eps':{1:-1.5459, 2:-1.3181},
                   'apf2nWpm2psr':{
                                   1:{4:-347.92, 5:-305.81, 6:-369.32, 7:-333.67, 8:-314.33},
                                   2:{4:-117.69, 5:-116.20, 6:-118.79, 7:-127.43, 8:-117.96},
                                  }
                  }

field_data_dict = {'nfr':{4:25, 5:10, 6:30, 7:29, 8:26},
                   'cbmean':{4:1.2377e+03, 5:815.6128, 6:834.0211, 7:731.1761, 8:607.0531},
                   'psmean':{4:2.3439, 5:2.4844, 6:2.4472, 7:2.5742, 8:2.5673}
                  }

class band_info:
    def __init__(self, inst):
        name_dict = {1:'I', 2:'H'}
        wl_dict = {1:1.05, 2:1.79}

        self.inst = inst
        self.name = name_dict[inst]
        self.wl = wl_dict[inst]

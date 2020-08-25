from ciber_paths import *
from utils_plotting import *
from astropy import units as u

fieldnamedict = {4:'elat10',
                 5:'elat30',
                 6:'BootesB',
                 7:'BootesA',
                 8:'SWIRE'}


field_center_dict = {# (approximately) the ra dec of the field centre [deg]
                     4: (191.5, 8.25),
                     5: (193.943, 27.998),
                     6: (218.109, 33.175),
                     7: (219.249, 34.832),
                     8: (241.530, 54.767)
                    }

magbindict = {'m_min':[16,17,18,19],
              'm_max':[17,18,19,20],
              # From MICECAT
              'zmean': [0.21, 0.24, 0.32, 0.41],
              'zmed': [0.21, 0.24, 0.29, 0.41]
             }

cal_factor_dict = {'framerate': 1/(1.6e-6 * 512 * 513 * 68 / 16), # 0.5599 frame / s
                   'apf2eps':{1:-1.5459, 2:-1.3181},
                   'apf2nWpm2psr':{
                                   # 1:{4:-347.92, 5:-305.81, 6:-369.32, 7:-333.67, 8:-314.33},
                                   # 2:{4:-117.69, 5:-116.20, 6:-118.79, 7:-127.43, 8:-117.96},
                                   1:{4:-448.60, 5:-333.58, 6:-461.59, 7:-339.27, 8:-361.43},
                                   2:{4:-122.28, 5:-118.09, 6:-137.84, 7:-111.95, 8:-115.39},
                                  }
                  }

filt_order_dict = {1:3, 2:5}

PSF_model_dict = {
                    1:{
                        4: (1.845e+00, 5.797e+00, 8.206e-03),
                        5: (1.542e+00, 4.619e+00, 9.598e-03),
                        6: (1.525e+00, 4.299e+00, 1.086e-02),
                        7: (1.529e+00, 4.422e+00, 1.031e-02),
                        8: (1.593e+00, 4.781e+00, 9.477e-03)
                      },

                    2:{
                        4: (1.478e+00, 5.284e+00, 6.797e-03),
                        5: (1.393e+00, 4.835e+00, 7.271e-03),
                        6: (1.475e+00, 5.181e+00, 7.045e-03),
                        7: (1.517e+00, 5.342e+00, 6.967e-03),
                        8: (1.504e+00, 5.282e+00, 7.024e-03)
                      }
                 }

# field_data_dict = {'nfr':{4:25, 5:10, 6:30, 7:29, 8:26},
#                    'cbmean':{4:1.2377e+03, 5:815.6128, 6:834.0211, 7:731.1761, 8:607.0531},
#                    'psmean':{4:2.3439, 5:2.4844, 6:2.4472, 7:2.5742, 8:2.5673}
#                   }

class band_info:
    def __init__(self, inst):
        name_dict = {1:'I', 2:'H'}
        wl_dict = {1:1.05, 2:1.79}

        self.inst = inst
        self.name = name_dict[inst]
        self.wl = wl_dict[inst]
        self.nu  = (self.wl * u.um.to(u.GHz, equivalencies=u.spectral()))
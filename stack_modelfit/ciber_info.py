mypaths = {'alldat':'/Users/ytcheng/ciber/doc/20170325_alldat/'}

fieldnamedict = {4:'elat10',
                 5:'elat30',
                 6:'BootesB',
                 7:'BootesA',
                 8:'SWIRE'}

class band_info:
    def __init__(self, inst):
        name_dict = {1:'I', 2:'H'}
        wl_dict = {1:1.05, 2:1.79}

        self.inst = inst
        self.name = name_dict[inst]
        self.wl = wl_dict[inst]

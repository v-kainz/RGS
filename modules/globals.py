# All constants and default values are collected here
import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator

# names 
names = {'deaf':'00_deaf', 'naive':'01_naive', 'uncritical+honest':'02_uncritical+honest', 'uncritical':'03_uncritical', 'ordinary':'04_ordinary',\
         'strategic':'05_strategic', 'egocentric':'06_egocentric', 'deceptive':'07_deceptive', 'flattering':'08_flattering', \
         'aggressive':'09_aggressive', 'shameless':'10_shameless', 'smart':'11_smart', 'Smart':'11a_Smart', 'SMart':'11b_SMart', 'clever':'12_clever', 'manipulative':'13_manipulative', \
         'dominant':'14_dominant', 'destructive':'15_destructive', 'good':'16_good', 'antistrategic': '17_antistrategic', \
         'MoLG': '18_MoLG'}

def init_LUT(load=True):
    global LUT

    class LookUpTable():
        def __init__(self, x, y, mu_data, la_data):
            # Replacing interp2d with RegularGridInterpolator
            self.mu = RegularGridInterpolator((x, y), mu_data, method='linear', bounds_error=False, fill_value=None)
            self.la = RegularGridInterpolator((x, y), la_data, method='linear', bounds_error=False, fill_value=None)
            
            if len(x) == len(y):
                self.len = len(x)
            else:
                raise ValueError('LUT is not quadratic. This is not implemented yet!')

    if load:
        filename_mu = 'LookUpTable/LUT_mu.json'
        filename_la = 'LookUpTable/LUT_la.json'
        length = 1000

        with open(filename_mu, 'r') as f:
            data_dict = json.loads(f.readline())
            mu_values = data_dict['data']
        with open(filename_la, 'r') as f:
            data_dict = json.loads(f.readline())
            la_values = data_dict['data']
        
        # interpolation setup
        x = y = np.linspace(0, length-1, length)  # index coordinate system
        LUT = LookUpTable(x, y, np.log10(np.array(mu_values)+1), np.log10(np.array(la_values)+1))

    else:
        LUT = None


global event_buffer
event_buffer = []
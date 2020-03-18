import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge
from pymms.pymms import MrMMS_SDC_API as sdc_api
import csv
from spacepy import pycdf
import pdb

def read_selections():
    file = '/Users/argall/Google Drive/Work/Papers/Current/Machine Learning/Priatt/selections.csv'
    
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)
        selections = []
        for idx, row in enumerate(csv_reader):
            try:
                start_datetime = np.datetime64('{2}-{1}-{0}T{3}'.format(*row[1][0:10].split('/'), row[1][11:]))
                end_datetime = np.datetime64('{2}-{1}-{0}T{3}'.format(*row[2][0:10].split('/'), row[2][11:]))
            
                selections.append((start_datetime, end_datetime))
            except:
                pass
    
    return selections


def get_ephemeris():
    selections = read_selections()
    sdc = sdc_api('mms1', 'mec', 'srvy', 'l2')
    pos_vname = 'mms1_mec_r_gse'
    RE = 6378
    
    r = np.empty([0,3])
    for t in selections:
        sdc.start_date = str(t[0])
        sdc.end_date  = str(t[1])
        mec_file = sdc.Download()
        
        try:
            with pycdf.CDF(mec_file[0]) as cdf:
                pos = cdf[pos_vname][...] / RE
                t_vname = cdf[pos_vname].attrs['DEPEND_0']
                time = cdf[t_vname][...]
                
                r = np.append(r, [np.mean(pos[(time >= t[0]) & (time <= t[1])], axis=0)], axis=0)
        except IndexError:
            print('{} files from {} to {}'.format(len(mec_file), t[0], t[1]))
            pass
    
    return r
        

def mpmodel(imf_bz=-0.595, Dp=1.915):

    # Magnetopause standoff distance and magnetotail flaring angle
    alpha = (0.58-0.010*imf_bz)*(1.0+0.010*Dp)
    if imf_bz > 0:
        r0 = (11.4+0.013*imf_bz)*Dp**(-1/6.6)
    else:
        r0 = (11.4+0.14*imf_bz)*Dp**(-1/6.6)
    
    # Angle between Earth-sun line and r
    theta = np.deg2rad(np.arange(-150, 150, 0.1))
    r = r0*(2/(1+np.cos(theta)))**alpha
    x = r*np.cos(theta) #r*(2/(r/r0)**(1/alpha) - 1)
    y = np.sqrt(r**2 - x**2)
    z = 0
    R = np.sqrt(y**2 + z**2)
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
#    fig.set_size_inches(8.5, 4.0, forward=True)
    
    # Plot R-x plane
    axes[0].plot(x, R, 'k')
    axes[0].set_xlim(-40, 25)
    axes[0].set_xlabel('X ($R_{E}$)')
    axes[0].set_ylim(0, 40)
    axes[0].set_ylabel('R ($R_{E}$)')
    
    # Add earth
    w_day = Wedge(0, 1, 0, 90, fc='white', edgecolor='black')
    w_night = Wedge(0, 1, 90, 180, fc='black', edgecolor='black')
    for w in [w_day, w_night]:
        axes[0].add_artist(w)
    
    # Plot xy-plane
    axes[1].plot(x, y, 'k', x, -y, 'k')
    axes[1].set_xlim(-40, 25)
    axes[1].set_xlabel('X ($R_{E}$)')
    axes[1].set_ylim(-30, 30)
    axes[1].set_ylabel('Y ($R_{E}$)')
    
    # Add earth
    w_day = Wedge(0, 1, -90, 90, fc='white', edgecolor='black')
    w_night = Wedge(0, 1, 90, -90, fc='black', edgecolor='black')
    for w in [w_day, w_night]:
        axes[1].add_artist(w)
    
    # Add magnetopause crossings
    mms_pos = get_ephemeris()
    
    pdb.set_trace()
    axes[0].plot(mms_pos[:,0], np.sqrt(mms_pos[:,1]**2 + mms_pos[:,2]**2), 'o', color='blue')
    axes[1].plot(mms_pos[:,0], mms_pos[:,1], 'o', color='blue')
    
    plt.show()

if __name__ == 'main':
    mpmodel()
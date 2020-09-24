from pymms.sdc import mrmms_sdc_api as api
from metaarray import metaarray, metabase
from . import fgm, util
import datetime as dt
import numpy as np
import xarray as xr
from scipy import constants
from matplotlib import pyplot as plt
from matplotlib import rc
import warnings

def check_spacecraft(sc):
    if sc not in ('mms1', 'mms2', 'mms3', 'mms4'):
        raise ValueError('{} is not a recongized SC ID. '
                         'Must be ("mms1", "mms2", "mms3", "mms4")'
                         .format(sc))

def check_mode(mode):
    
    modes = ('brst', 'fast')
    if mode == 'srvy':
        mode = 'fast'
    
    if mode not in modes:
        raise ValueError('Mode "{0}" is not in {1}'.format(mode, modes))

    return mode


def check_species(species):
    if species not in ('e', 'i'):
        raise ValueError('{} is not a recongized species. '
                         'Must be ("i", "e")')


def compare_moments_xarray():
    sc = 'mms1'
    mode = 'fast'
    species = 'i'
    start_date = dt.datetime(2020, 4, 3, 22, 00, 23)
    end_date = dt.datetime(2020, 4, 3, 23, 12, 13)
    
    # Read the data
    moms_df = load_moms(sc, mode, species, start_date, end_date)
    dist_xr = load_dist_xarray(sc, mode, species, start_date, end_date)
    max_xr = max_dist_xarray(dist_xr,
                             moms_df['N{}'.format(species)],
                             moms_df['V{}'.format(species)],
                             moms_df['t{}'.format(species)])
    
    # Density
    ni_xr = density_xarray(dist_xr)
    ni_max_dist = density_xarray(max_xr)
    
    # Entropy
    s_xr = entropy_xarray(dist_xr)
    s_max_dist = entropy_xarray(ni_max_dist)
    s_max = max_entropy_pd(moms_df['N{}'.format(species)],
                           moms_df['p{}'.format(species)])
    
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, squeeze=False)
    
    # Denisty
    moms_df['N{}'.format(species)].plot(ax=axes[0,0])
    ni_xr.plot(ax=axes[0,0])
    ni_max_dist(ax=axes[0,0])
    
    # Entropy
    s_xr.plot(ax=axes[1,0])
    s_max_dist.plot(ax=axes[1,0])
    s_max.plot(ax=axes[1,0])

    return fig, axes


def density_xarray(dist, **kwargs):
    
    mass = species_to_mass(dist.attrs['species'])
    
    f = precondition_xarray(dist, **kwargs)
    
    N = f.integrate('phi')
    N = (np.sin(N['theta'])*N).integrate('theta')
    
    # Integrate over Energy
    E0 = 100
    y = np.sqrt(f['U']) / (1-f['U'])**(5/2) * N
    y[:,-1] = 0
    N = (1e6 * np.sqrt(2) * (constants.eV * E0 / mass)**(3/2)
         * np.trapz(y, y['U'], axis=y.get_axis_num('e-bin'))
         )
    
    N = xr.DataArray(N, dims='time', coords={'time': dist['time']})
    
    # Add metadata
    N.name = 'N{}'.format(N.attrs['species'])
    N.attrs['long_name'] = ('Number density calculated by integrating the '
                            'distribution function.')
    N.attrs['standard_name'] = 'number_density'
    N.attrs['units'] = 'cm^-3'
    
    return N


def entropy_xarray(dist, species, scpot=None):
    
    kB = constants.k # J/K
    mass = species_to_mass(species)
    
    f = precondition_xarray(dist, **kwargs)
    S = 1e12 * f.copy()
    S[S == 0] = 1
    S = (S * np.log(S)).integrate('phi')
    S = (np.sin(S['theta']) * S).integrate('theta')
    
    # Integrate over Energy
    E0 = 100
    y = np.sqrt(S['U']) / (1 - S['U'])**(5/2) * S
    y[:,-1] = 0
    S = (-kB * np.sqrt(2) * (constants.eV * E0 / mass)**(3/2)
         * np.trapz(y, y['U'], axis=y.get_axis_num('e-bin'))
         )
    
    S = xr.DataArray(S, dims='time', coords={'time': dist['time']})
    
    S.name = 'S{}'.format(S.attrs['species'])
    S.attrs['long_name'] = 'Velocity space entropy density'
    S.attrs['standard_name'] = 'entropy_density'
    S.attrs['units'] = 'J/K/m^3 ln(s^3/m^6)'
    
    return S


def load_dist_xarray(sc='mms1', mode='fast', species='i',
                     start_date=dt.datetime(2017, 11, 24, 0),
                     end_date=dt.datetime(2017, 11, 24, 23, 59, 59)):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    species : str
        Particle species: ('i', 'e') for ions and electrons, respectively.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_species(species)
    
    # File and variable name parameters
    instr = 'd{0}s'.format(species)
    optdesc = instr+'-dist'
    fpi_dist_vname = '_'.join((sc, instr, 'dist', mode))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, 'fpi', mode, 'l2',
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    fpi_files = sdc.download_files()
    
    # Read the data from files
    fpi_ds = util.cdf_to_ds(fpi_files[0], fpi_dist_vname)
    
    '''
    # Read the data from files
    dist = metaarray.from_cdflib(fpi_files, fpi_dist_vname,
                                 start_date=start_date,
                                 end_date=end_date)
    
    xr_dist = xr.DataArray(dist,
                           dims=('time', 'phi', 'theta', 'e-bin'),
                           coords={'time': dist.x0, 
                                   'phi': dist.x1,
                                   'theta': dist.x2,
                                   'energy': (('time', 'e-bin'), dist.x3)
                                   }
                           )
    
    xr_dist.attrs['species'] = species
    '''
    
    return fpi_ds


def load_moms_xarray(sc, mode, species,
                     start_date, end_date,
                     maxwell_entropy=False):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    species : str
        Particle species: ('i', 'e') for ions and electrons, respectively.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_species(species)
    
    # File and variable name parameters
    instr = 'd{0}s'.format(species)
    optdesc = instr+'-moms'
    n_vname = '_'.join((sc, instr, 'numberdensity', mode))
    v_vname = '_'.join((sc, instr, 'bulkv', 'dbcs', mode))
    p_vname = '_'.join((sc, instr, 'prestensor', 'dbcs', mode))
    t_vname = '_'.join((sc, instr, 'temptensor', 'dbcs', mode))
    q_vname = '_'.join((sc, instr, 'heatq', 'dbcs', mode))
    t_para_vname = '_'.join((sc, instr, 'temppara', mode))
    t_perp_vname = '_'.join((sc, instr, 'tempperp', mode))
    varnames = [n_vname, v_vname, p_vname, t_vname, q_vname,
                t_para_vname, t_perp_vname]
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, 'fpi', mode, 'l2',
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    fpi_files = sdc.download_files()
    
    # Read the data from files
    fpi_ds = util.cdf_to_ds(fpi_files, varnames)
    
    return fpi_ds


def max_dist_xarray(dist, density, bulkv, temperature):
    
    eV2K = constants.value('electron volt-kelvin relationship')
    eV2J = constants.eV
    kB   = constants.k
    mass = species_to_mass(dist.attrs['species'])
    
    phi = np.deg2rad(dist['phi'])
    theta = np.deg2rad(dist['theta'])
    velocity = np.sqrt(2.0 * eV2J / mass * dist['energy'])  # m/s
    
    f_out = xr.DataArray(np.zeros(dist.shape, dtype='float32'),
                         dims=dist.dims,
                         coords=dist.coords)
    
    for i, p in enumerate(phi):
        for j, t in enumerate(theta):
            for k in range(velocity.shape[-1]):
                v = velocity[:,k]
                vxsqr = (-v * np.sin(t) * np.cos(p) - (1e3*bulkv[:,0]))**2
                vysqr = (-v * np.sin(t) * np.sin(p) - (1e3*bulkv[:,1]))**2
                vzsqr = (-v * np.cos(t) - (1e3*bulkv[:,2]))**2
                
                f_out[:,i,j,k] = (1e-6 * density 
                                  * (mass / (2 * np.pi * kB * eV2K * temperature))**(3.0/2.0)
                                  * np.exp(-mass * (vxsqr + vysqr + vzsqr)
                                        / (2.0 * kB * eV2K * temperature))
                                  )
    
    f_out.name = 'Equivalent Maxwellian distribution'
    f_out.attrs['species'] = dist.attrs['species']
    f_out.attrs['long_name'] = ('Maxwellian distribution constructed from '
                                'the density, velocity, and temperature of '
                                'the measured distribution function.')
    f_out.attrs['standard_name'] = 'maxwellian_distribution'
    f_out.attrs['units'] = 's^3/cm^6'
    
    return f_out


def max_entropy_pd(N, P):
    J2eV = constants.value('joule-electron volt relationship')
    kB   = constants.k
    mass = species_to_mass(N.attrs['species'])
    
    Sb = (-kB * 1e6 * N
          * (np.log((1e19 * mass * N**(5.0/3.0)
                    / 2 / np.pi / P)**(3/2)
                   )
             - 3/2
             )
          )
    
    Sb.name = 'S{}'.format(N.attrs['species'])
    Sb.species = N.attrs['species']
    Sb.attrs['long_name'] = 'Boltzmann entropy for a given density and pressure.'
    Sb.attrs['standard_name'] = 'Boltzmann_entropy'
    Sb.attrs['units'] = 'J/K/m^3 ln(s^3/m^6)'
    return Sb


def precondition_xarray(dist, E0=100, E_low=10, scpot=None,
                        low_energy_extrapolation=True,
                        high_energy_extrapolation=True):
    '''
    Before being sent to the integration routine, skymaps are preprocessed
    in the following manner:
      1. f(phi = 0) is repeated as f(phi = 360) to ensure that the periodic
         boundary condition is incorporated to the azimuthal integration.
      2. f(theta=0) = 0 and f(theta=180) = 0 data points are added to ensure
         the polar integration goes from 0 to 180.  The sin(theta)
         dependence of the polar integration force the integrand at
         theta = 0 and theta = 180 to zero regardless of the value of the
         phase space density
      3. f(U = 0) = 0 and f(U=1) =0 data points are added to ensure the
         integration goes from E->0 to E->infinity. V = 0 forces the
         integrand equal to zero regardless of the phase space density. 
    
    Parameters
    ----------
    dist : `metaarray.MetaArray`
        The velocity distribution function (s^3/cm^6) with azimuth, polar,
        and energy dependencies as attributes.
    E0 : float
        Energy value (eV) used when mapping energy bins from range [0,Emax]
        to [0, inf)
    E_low : float
        Energy value (eV) representing the low-energy cut-off
    '''
    
    # pad looks like right approach, but it is still experimental and
    # instead of using the values specified by the constant_values keyword,
    # it always uses np.nan.
    '''
    out = dist.pad(pad_width={'phi', (0, 1),
                              'theta', (1, 1),
                              'e-bin', (1, 1)},
                   mode='constant',
                   constant_values={'phi': (0, 0),
                                    'theta': (0, 0),
                                    'e-bin': (0, 0)}
                   )
    '''
    
    # Append boundary point to make phi periodic
    f_phi = (dist[:,0,:,:].assign_coords(phi=dist['phi'][0] + 360.0))
    f_out = xr.concat([dist, f_phi], 'phi')

    # Create boundary points to have theta range be [0,180] inclusive.
    # Note that the sin(theta) forces the integrand to be 0 at the
    # boundaries regardless of what the distribution function
    f_theta = xr.DataArray(np.zeros(shape=(2,)),
                           dims='theta',
                           coords={'theta': [0, 180]})
    
    # Append the boundary points to the beginning and end of the
    # array. This will change the order of the dimensions. Set the
    # values at the boundaries to zero (arbitrary) and transpose
    # back to the original shape.
    f_out = xr.concat([f_theta[0], f_out], 'theta')
    f_out = xr.concat([f_out, f_theta[1]], 'theta')

    # This throws an error:
    # ValueError: The truth value of an array with more than one element
    #             is ambiguous. Use a.any() or a.all()
    #
    # f_out = xr.combine_by_coords(f_out, f_theta)
    
    # Adjust for spacecraft potential
    if scpot is not None:
        pass
    
    iGtELow = ((f_out['energy'][0, :] >= E_low) == False).argmin().values.item()
    f_out = f_out[:, :, :, iGtELow:]
    U = f_out['energy'] / (f_out['energy'] + E0)
    U = U.drop_vars('energy')
    U_boundaries = xr.DataArray(np.zeros(shape=(f_out.sizes['time'], 2)),
                                dims=('time', 'e-bin'),
                                coords={'time': f_out['time']}
                                )
    U_boundaries[:,-1] = 1.0
    
    # Append the boundary points to the beginning and end of the array.
    U = xr.concat([U_boundaries[:,0], U], 'e-bin')
    U = xr.concat([U, U_boundaries[:,-1]], 'e-bin')
    
    # Create boundary points for the energy at 0 and infinity, essentially
    # extrapolating the distribution to physical limits. Since absolute
    # zero and infinite energies are impossible, set the values in the
    # distribution to zero at those points. This changes the order of the
    # dimensions so they will have to be transposed back.
    f_energy = xr.DataArray(np.zeros((2,)),
                            dims='e-bin',
                            coords={'energy': ('e-bin', [0, np.inf])})
    
    # Append the extrapolated points to the distribution
    f_out = xr.concat([f_energy[0], f_out], 'e-bin')
    f_out = xr.concat([f_out, f_energy[1]], 'e-bin')
    
    # Assign U as another coordinate
    f_out = f_out.assign_coords(U=U)

    # Convert to radians
    f_out = f_out.assign_coords(phi=np.deg2rad(f_out['phi']))
    f_out = f_out.assign_coords(theta=np.deg2rad(f_out['theta']))
    return f_out


def compare_moments():
    sc = 'mms1'
    mode = 'fast'
    species = 'i'
    start_date = dt.datetime(2020, 4, 3, 22, 00, 23)
    end_date = dt.datetime(2020, 4, 3, 23, 12, 13)
    
    # Read the data
    moms = load_moms(sc, mode, species, start_date, end_date)
#    scpot = load_scpot(sc, mode, start_date, end_date)
    dist = load_dist(sc, mode, species, start_date, end_date)
    
    # Create a Maxwellian distribution
    t_moms = moms[3]
    t_moms = (t_moms[:,0,0] + t_moms[:,1,1] + t_moms[:,2,2]) / 3.0
    t_moms.x0 = moms[3].x0
    maxdist = maxwellian_distribution(dist, moms[0], moms[1], t_moms,
                                      species=species)
    
    # Scalar pressure
    p_moms = moms[2]
    p_moms = (p_moms[:,0,0] + p_moms[:,1,1] + p_moms[:,2,2]) / 3.0
    p_moms.x0 = moms[2].x0
    
    # Density
    n_moms = moms[0]
    n_dist = density(dist, species)
    n_max_dist = density(maxdist, species)
    
    n_moms.label = '$N_{i,moms}$'
    n_moms.title = '$N_{i}$\n($cm^{-3}$)'
    n_moms.plot_title = ''
    
    n_dist.label = '$N_{i,dist}$'
    n_dist.plot_title = ''
    
    n_max_dist.label = '$N_{i,max}$'
    n_max_dist.plot_title = ''
    
    # Entropy
    S_dist = entropy(dist, species)
    S_max_dist = entropy(maxdist, species)
    S_max_moms = maxwellian_entropy(n_moms, p_moms, species)
    
    S_dist.label = '$S_{dist}$'
    S_max_dist.label = '$S_{max,dist}$'
    S_max_moms.label = '$S_{max,moms}$'
    
    # Bulk velocity
    vx_moms = moms[1][:,0]
    vy_moms = moms[1][:,1]
    vz_moms = moms[1][:,2]
    vx_dist, vy_dist, vz_dist = velocity(dist, species, N=n_dist)
    vx_max_dist, vy_max_dist, vz_max_dist = velocity(maxdist, species,
                                                     N=n_max_dist,
                                                     precondition=True)
    v_dist = np.concatenate((vx_dist[:, np.newaxis],
                             vy_dist[:, np.newaxis],
                             vz_dist[:, np.newaxis]),
                            axis=1)
    v_max_dist = np.concatenate((vx_max_dist[:, np.newaxis],
                                 vy_max_dist[:, np.newaxis],
                                 vz_max_dist[:, np.newaxis]),
                                axis=1)
    
    vx_moms.label = '$V_{x,moms}$'
    vx_dist.label = '$V_{x,dist}$'
    vx_max_dist.label = '$V_{x,max}$'
    
    vy_moms.label = '$V_{y,moms}$'
    vy_dist.label = '$V_{y,dist}$'
    vy_max_dist.label = '$V_{y,max}$'
    
    vz_moms.label = '$V_{z,moms}$'
    vz_dist.label = '$V_{z,dist}$'
    vz_max_dist.label = '$V_{z,max}$'
    
    # Temperature
    T_moms = moms[3]
    T_dist = temperature(dist, species, N=n_dist, V=v_dist)
    T_max_dist = temperature(maxdist, species, N=n_max_dist, V=v_max_dist)
    
    Txx_moms = T_moms[:,0,0]
    Txx_moms.label = '$T_{xx,moms}$'
    Txx_moms.plot_title = ''
    Txx_moms.title = '$T_{{{0},xx}}$\n(eV)'.format(species)
    Txx_dist = T_dist[:,0,0]
    Txx_dist.label = '$T_{xx,dist}$'
    Txx_dist.plot_title = ''
    Txx_dist.title = '$T_{{{0},xx}}$\n(eV)'.format(species)
    Txx_max_dist = T_max_dist[:,0,0]
    Txx_max_dist.label = '$T_{xx,max}$'
    Txx_max_dist.plot_title = ''
    Txx_max_dist.title = '$T_{{{0},xx}}$\n(eV)'.format(species)
    
    Tyy_moms = T_moms[:,1,1]
    Tyy_moms.label = '$T_{yy,moms}$'
    Tyy_moms.title = '$T_{{{0},yy}}$\n(eV)'.format(species)
    Tyy_dist = T_dist[:,1,1]
    Tyy_dist.label = '$T_{yy,dist}$'
    Tyy_dist.title = '$T_{{{0},yy}}$\n(eV)'.format(species)
    Tyy_max_dist = T_max_dist[:,1,1]
    Tyy_max_dist.label = '$T_{yy,max}$'
    Tyy_max_dist.title = '$T_{{{0},yy}}$\n(eV)'.format(species)
    
    Tzz_moms = T_moms[:,2,2]
    Tzz_moms.label = '$T_{zz,moms}$'
    Tzz_moms.title = '$T_{{{0},zz}}$\n(eV)'.format(species)
    Tzz_dist = T_dist[:,2,2]
    Tzz_dist.label = '$T_{zz,dist}$'
    Tzz_dist.title = '$T_{{{0},zz}}$\n(eV)'.format(species)
    Tzz_max_dist = T_max_dist[:,2,2]
    Tzz_max_dist.label = '$T_{zz,max}$'
    Tzz_max_dist.title = '$T_{{{0},zz}}$\n(eV)'.format(species)
    
    Txy_moms = T_moms[:,0,1]
    Txy_moms.label = '$T_{xy,moms}$'
    Txy_moms.title = '$T_{{{0},xy}}$\n(eV)'.format(species)
    Txy_dist = T_dist[:,0,1]
    Txy_dist.label = '$T_{xy,dist}$'
    Txy_dist.title = '$T_{{{0},xy}}$\n(eV)'.format(species)
    Txy_max_dist = T_max_dist[:,0,1]
    Txy_max_dist.label = '$T_{xy,max}$'
    Txy_max_dist.title = '$T_{{{0},xy}}$\n(eV)'.format(species)
    
    Txz_moms = T_moms[:,0,2]
    Txz_moms.label = '$T_{xz,moms}$'
    Txz_moms.title = '$T_{{{0},xz}}$\n(eV)'.format(species)
    Txz_dist = T_dist[:,0,2]
    Txz_dist.label = '$T_{xz,dist}$'
    Txz_dist.title = '$T_{{{0},xz}}$\n(eV)'.format(species)
    Txz_max_dist = T_max_dist[:,0,2]
    Txz_max_dist.label = '$T_{xz,max}$'
    Txz_max_dist.title = '$T_{{{0},xz}}$\n(eV)'.format(species)
    
    Tyz_moms = T_moms[:,1,2]
    Tyz_moms.label = '$T_{yz,moms}$'
    Tyz_moms.title = '$T_{{{0},yz}}$\n(eV)'.format(species)
    Tyz_dist = T_dist[:,1,2]
    Tyz_dist.label = '$T_{yz,dist}$'
    Tyz_dist.title = '$T_{{{0},yz}}$\n(eV)'.format(species)
    Tyz_max_dist = T_max_dist[:,1,2]
    Tyz_max_dist.label = '$T_{yz,max}$'
    Tyz_max_dist.title = '$T_{{{0},yz}}$\n(eV)'.format(species)
    
    # Scalar temperature
    t_dist = (T_dist[:,0,0] + T_dist[:,1,1] + T_dist[:,2,2]) / 3
    t_max_dist = (T_max_dist[:,0,0] + T_max_dist[:,1,1] + T_max_dist[:,2,2]) / 3
    
    t_dist.x0 = T_dist.x0
    t_max_dist.x0 = T_dist.x0
    
    t_moms.title = 'T\n(eV)'
    t_moms.label = '$T_{moms}$'
    t_dist.label = '$T_{dist}$'
    t_max_dist.label = '$T_{max}$'
    
    # Pressure
    P_moms = moms[2]
    P_dist = pressure(dist, species, N=n_dist, T=T_dist)
    P_max_dist = pressure(dist, species, N=n_max_dist, T=T_max_dist)
    
    Pxx_moms = P_moms[:,0,0]
    Pxx_moms.label = '$P_{xx,moms}$'
    Pxx_moms.plot_title = ''
    Pxx_moms.title = '$P_{{{0},xx}}$\n(nPa)'.format(species)
    Pxx_dist = P_dist[:,0,0]
    Pxx_dist.label = '$P_{xx,dist}$'
    Pxx_dist.plot_title = ''
    Pxx_dist.title = '$P_{{{0},xx}}$\n(nPa)'.format(species)
    Pxx_max_dist = P_max_dist[:,0,0]
    Pxx_max_dist.label = '$P_{xx,max}$'
    Pxx_max_dist.plot_title = ''
    Pxx_max_dist.title = '$P_{{{0},xx}}$\n(nPa)'.format(species)
    
    Pyy_moms = P_moms[:,1,1]
    Pyy_moms.label = '$P_{yy,moms}$'
    Pyy_moms.title = '$P_{{{0},yy}}$\n(nPa)'.format(species)
    Pyy_dist = P_dist[:,1,1]
    Pyy_dist.label = '$P_{yy,dist}$'
    Pyy_dist.title = '$P_{{{0},yy}}$\n(nPa)'.format(species)
    Pyy_max_dist = P_max_dist[:,1,1]
    Pyy_max_dist.label = '$P_{yy,max}$'
    Pyy_max_dist.title = '$P_{{{0},yy}}$\n(nPa)'.format(species)
    
    Pzz_moms = P_moms[:,2,2]
    Pzz_moms.label = '$P_{zz,moms}$'
    Pzz_moms.title = '$P_{{{0},zz}}$\n(nPa)'.format(species)
    Pzz_dist = P_dist[:,2,2]
    Pzz_dist.label = '$P_{zz,dist}$'
    Pzz_dist.title = '$P_{{{0},zz}}$\n(nPa)'.format(species)
    Pzz_max_dist = P_max_dist[:,2,2]
    Pzz_max_dist.label = '$P_{zz,max}$'
    Pzz_max_dist.title = '$P_{{{0},zz}}$\n(nPa)'.format(species)
    
    Pxy_moms = P_moms[:,0,1]
    Pxy_moms.label = '$P_{xy,moms}$'
    Pxy_moms.title = '$P_{{{0},xy}}$\n(nPa)'.format(species)
    Pxy_dist = P_dist[:,0,1]
    Pxy_dist.label = '$P_{xy,dist}$'
    Pxy_dist.title = '$P_{{{0},xy}}$\n(nPa)'.format(species)
    Pxy_max_dist = P_max_dist[:,0,1]
    Pxy_max_dist.label = '$P_{xy,max}$'
    Pxy_max_dist.title = '$P_{{{0},xy}}$\n(nPa)'.format(species)
    
    Pxz_moms = P_moms[:,0,2]
    Pxz_moms.label = '$P_{xz,moms}$'
    Pxz_moms.title = '$P_{{{0},xz}}$\n(nPa)'.format(species)
    Pxz_dist = P_dist[:,0,2]
    Pxz_dist.label = '$P_{xz,dist}$'
    Pxz_dist.title = '$P_{{{0},xz}}$\n(nPa)'.format(species)
    Pxz_max_dist = P_max_dist[:,0,2]
    Pxz_max_dist.label = '$P_{xz,max}$'
    Pxz_max_dist.title = '$P_{{{0},xz}}$\n(nPa)'.format(species)
    
    Pyz_moms = P_moms[:,1,2]
    Pyz_moms.label = '$P_{yz,moms}$'
    Pyz_moms.title = '$P_{{{0},yz}}$\n(nPa)'.format(species)
    Pyz_dist = P_dist[:,1,2]
    Pyz_dist.label = '$P_{yz,dist}$'
    Pyz_dist.title = '$P_{{{0},yz}}$\n(nPa)'.format(species)
    Pyz_max_dist = P_max_dist[:,1,2]
    Pyz_max_dist.label = '$P_{yz,max}$'
    Pyz_max_dist.title = '$P_{{{0},yz}}$\n(nPa)'.format(species)
    
    # Scalar pressure
    p_dist = (P_dist[:,0,0] + P_dist[:,1,1] + P_dist[:,2,2]) / 3.0
    p_max_dist = (P_max_dist[:,0,0] + P_max_dist[:,1,1] + P_max_dist[:,2,2]) / 3.0
    
    p_dist.x0 = P_dist.x0
    p_max_dist.x0 = P_max_dist.x0
    
    fig, axes = metabase.MetaCache.plot(
                    [[n_moms, n_dist, n_max_dist],
                     [S_dist, S_max_dist, S_max_moms],
                     [vx_moms, vx_dist, vx_max_dist],
                     [vy_moms, vy_dist, vy_max_dist],
                     [vz_moms, vz_dist, vz_max_dist],
                     [t_moms, t_dist, t_max_dist],
                     [Txx_moms, Txx_dist, Txx_max_dist],
                     [Tyy_moms, Tyy_dist, Tyy_max_dist],
                     [Tzz_moms, Tzz_dist, Tzz_max_dist],
                     [Txy_moms, Txy_dist, Txy_max_dist],
                     [Txz_moms, Txz_dist, Txz_max_dist],
                     [Tyz_moms, Tyz_dist, Tyz_max_dist],
                     [Pxx_moms, Pxx_dist, Pxx_max_dist],
                     [Pyy_moms, Pyy_dist, Pyy_max_dist],
                     [Pzz_moms, Pzz_dist, Pzz_max_dist],
                     [Pxy_moms, Pxy_dist, Pxy_max_dist],
                     [Pxz_moms, Pxz_dist, Pxz_max_dist],
                     [Pyz_moms, Pyz_dist, Pyz_max_dist]
                     ], nrows=6, ncols=3, figsize=(11, 7), legend=True
                    )
    
    fig.suptitle('Comparing FPI Moments, Integrated Distribution, Equivalent Maxwellian')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, wspace=0.7)
    return fig, axes


def density(dist, species, **kwargs):
    
    mass = species_to_mass(species)
    
    f_out, phi, theta, U = precondition(dist, **kwargs)
    
    # Integrate over phi
    N = int_trapezoid(phi, f_out)
    
    # Integrate over theta
    N = int_trapezoid(theta, np.sin(theta)*N)
    
    # Integrate over Energy
    E0 = 100
    y = np.divide(np.sqrt(U), (1-U)**(5/2)) * N
    y[:,-1] = 0
    N = (1e6 * np.sqrt(2) * (constants.eV * E0 / mass)**(3/2)
         * int_trapezoid(U, y)
         )
    
    # Add metadata
    N.name = 'D{0}S Density'.format(species.upper())
    N.plot_title = ('Density calculated from d{}s velocity distribution '
                    'function.'.format(species))
    N.label = '$N_{' + species + '}$'
    N.title = '$N_{' + species + '}$'
    N.scale = 'linear'
    N.x0 = dist.x0
    
    return N


def entropy(dist, species, scpot=None):
    
    kB = constants.k # J/K
    mass = species_to_mass(species)
    
    f_out, phi, theta, U = precondition(dist)
    
    # Integrate over phi
    #
    S = 1e12 * f_out.copy() # s^3 / m^6
    S[f_out == 0] = 1
    S = int_trapezoid(phi, S*np.log(S))
    
    # Integrate over theta
    S = int_trapezoid(theta, np.sin(theta)*S)
    
    # Integrate over Energy
    E0 = 100
    y = np.divide(np.sqrt(U), (1-U)**(5/2)) * S
    y[:,-1] = 0
    S = (-kB * np.sqrt(2) * (constants.eV * E0 / mass)**(3/2)
         * int_trapezoid(U, y)
         )
    
    # Add metadata
    S.name = 'D{0}S Entropy'.format(species.upper())
    S.plot_title = ('Entropy calculated from d{}s velocity distribution '
                    'function.'.format(species))
    S.label = '$S_{' + species + '}$'
    S.title = '$S_{' + species + '}$\n(J/K/$m^3$)'
    S.scale = 'linear'
    S.units = 'J/K/m^3 ln(s^3/m^6)'
    S.x0 = dist.x0
    
    return S


def load_dist(sc, mode, species, start_date, end_date):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    species : str
        Particle species: ('i', 'e') for ions and electrons, respectively.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_species(species)
    
    # File and variable name parameters
    instr = 'd{0}s'.format(species)
    optdesc = instr+'-dist'
    fpi_dist_vname = '_'.join((sc, instr, 'dist', mode))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, 'fpi', mode, 'l2',
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    fpi_files = sdc.download_files()
    
    # Read the data from files
    dist = metaarray.from_cdflib(fpi_files, fpi_dist_vname,
                                 start_date=start_date,
                                 end_date=end_date)
    
    return dist


def load_moms(sc, mode, species, start_date, end_date,
              maxwell_entropy=False):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    species : str
        Particle species: ('i', 'e') for ions and electrons, respectively.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_species(species)
    
    # File and variable name parameters
    instr = 'd{0}s'.format(species)
    optdesc = instr+'-moms'
    n_vname = '_'.join((sc, instr, 'numberdensity', mode))
    v_vname = '_'.join((sc, instr, 'bulkv', 'dbcs', mode))
    p_vname = '_'.join((sc, instr, 'prestensor', 'dbcs', mode))
    t_vname = '_'.join((sc, instr, 'temptensor', 'dbcs', mode))
    q_vname = '_'.join((sc, instr, 'heatq', 'dbcs', mode))
    t_para_vname = '_'.join((sc, instr, 'temppara', mode))
    t_perp_vname = '_'.join((sc, instr, 'tempperp', mode))
    varnames = [n_vname, v_vname, p_vname, t_vname, q_vname,
                t_para_vname, t_perp_vname]
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, 'fpi', mode, 'l2',
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    fpi_files = sdc.download_files()
    
    
    # Read the data from files
    fpi_df = util.read_cdf_vars(fpi_files, varnames)
#    data = []
#    for v in varnames:
#        var = metaarray.from_cdflib(fpi_files, v,
#                                    start_date=start_date,
#                                    end_date=end_date)
#        data.append(var)
    
    # Calculate Maxwellian Entropy
    if maxwell_entropy:
        data.append(mexwellian_entropy(data[0]))
    
    # Rename columns
    fpi_df.rename(columns={n_vname: 'Ni'}, inplace=True)
    fpi_df.rename(columns={t_para_vname: 'Ti_para'}, inplace=True)
    fpi_df.rename(columns={t_perp_vname: 'Ti_perp'}, inplace=True)
    util.rename_df_cols(fpi_df, v_vname, ('Vix', 'Viy', 'Viz'))
    util.rename_df_cols(fpi_df, q_vname, ('Qi_xx', 'Qi_yy', 'Qi_zz'))
    util.rename_df_cols(fpi_df, t_vname,
                        ('Ti_xx', 'Ti_xy', 'Ti_xz',
                         'Ti_yx', 'Ti_yy', 'Ti_yz',
                         'Ti_zx', 'Ti_zy', 'Ti_zz'
                         ))
    util.rename_df_cols(fpi_df, p_vname,
                        ('Pi_xx', 'Pi_xy', 'Pi_xz',
                         'Pi_yx', 'Pi_yy', 'Pi_yz',
                         'Pi_zx', 'Pi_zy', 'Pi_zz'
                         ))

    # Drop redundant components of the pressure and temperature tensors
    fpi_df.drop(columns=['Ti_yx', 'Ti_zx', 'Ti_zy',
                         'Pi_yx', 'Pi_zx', 'Pi_zy'],
                inplace=True
                )
    
    # Scalar temperature and pressure
    fpi_df['ti'] = (fpi_df['Ti_xx'] + fpi_df['Ti_yy'] + fpi_df['Ti_zz'])/3.0
    fpi_df['pi'] = (fpi_df['Pi_xx'] + fpi_df['Pi_yy'] + fpi_df['Pi_zz'])/3.0
    
    return fpi_df


def int_reimann(x, y):
    pass


def int_trapezoid(x, y):
    return 0.5 * np.sum((x[:,1:,...] - x[:,:-1,...])
                        * (y[:,1:,...] + y[:,:-1,...]),
                        axis=1
                        )


def maxwellian_distribution(dist, density, bulkv, temperature, species='i'):
    
    eV2K = constants.value('electron volt-kelvin relationship')
    eV2J = constants.eV
    kB   = constants.k
    mass = species_to_mass(species)
    
    phi = np.deg2rad(dist.x1)
    theta = np.deg2rad(dist.x2)
    energy = dist.x3
    velocity = np.sqrt(2.0 * eV2J / mass * energy)  # m/s
    
    ntime = len(density)
    nphi = len(phi)
    ntheta = len(theta)
    nenergy = energy.shape[-1]
    f_out = np.zeros((ntime, nphi, ntheta, nenergy), dtype='float32')
    
    for i, p in enumerate(phi):
        for j, t in enumerate(theta):
            for k in range(velocity.shape[-1]):
                v = velocity[:,k]
                vxsqr = (-v * np.sin(t) * np.cos(p) - (1e3*bulkv[:,0]))**2
                vysqr = (-v * np.sin(t) * np.sin(p) - (1e3*bulkv[:,1]))**2
                vzsqr = (-v * np.cos(t) - (1e3*bulkv[:,2]))**2
                
                f_out[:,i,j,k] = (1e-6 * density 
                                  * (mass / (2 * np.pi * kB * eV2K * temperature))**(3.0/2.0)
                                  * np.exp(-mass * (vxsqr + vysqr + vzsqr)
                                        / (2.0 * kB * eV2K * temperature))
                                  )
    
    f_out = metaarray.MetaArray(f_out)
    f_out.x0 = dist.x0
    f_out.x1 = dist.x1
    f_out.x2 = dist.x2
    f_out.x3 = dist.x3
    
    return f_out


def maxwellian_entropy(N, P, species='i'):
    J2eV = constants.value('joule-electron volt relationship')
    kB   = constants.k
    mass = species_to_mass(species)
    
    Sb = (-kB * 1e6 * N
          * (np.log((1e19 * mass * N**(5.0/3.0)
                    / 2 / np.pi / P)**(3/2)
                   )
             - 3/2
             )
          )
    
    Sb.x0 = N.x0
    Sb.units = 'J/K/m^3 ln(s^3/m^6)'
    return Sb


def moments(dist, species, moment, **kwargs):
    valid_moms = ('density', 'velocity', 'pressure', 'temperature',
                  'N', 'V', 'P', 'T')
    if moment not in valid_moms:
        raise ValueError('Moment {0} is not in {1}'
                         .format(moment, valid_moms)
                         )
    
    if moment in ('density', 'N'):
        func = density
    elif moment in ('velocity', 'V'):
        func = velocity
    elif moment in ('temperature', 'T'):
        func = temperature
    elif moment in ('pressure', 'P'):
        func = pressure
    
    return func(data, species, **kwargs)


def plot_entropy():
    sc = 'mms1'
    mode = 'fast'
    start_date = dt.datetime(2020, 4, 3, 22, 00, 23)
    end_date = dt.datetime(2020, 4, 3, 23, 12, 13)
    
    # Read the data
    b = fgm.load_data(sc, mode, start_date, end_date)
    dis_moms = load_moms(sc, mode, 'i', start_date, end_date)
    des_moms = load_moms(sc, mode, 'e', start_date, end_date)
    dis_dist = load_dist(sc, mode, 'i', start_date, end_date)
    des_dist = load_dist(sc, mode, 'e', start_date, end_date)
    
    # Magnetic field
    Bx = b[:,0]
    Bx.plot_title = ''
    Bx.title = 'B\n(nT)'
    Bx.label = '$B_{x}$'
    By = b[:,1]
    By.plot_title = ''
    By.title = 'B\n(nT)'
    By.label = '$B_{y}$'
    Bz = b[:,2]
    Bz.plot_title = ''
    Bz.title = 'B\n(nT)'
    Bz.label = '$B_{z}$'
    
    # Density
    ni_moms = dis_moms[0]
    ni_moms.label = '$N_{i}$'
    ni_moms.title = '$N$\n($cm^{-3}$)'
    ni_moms.plot_title = ''
    
    ne_moms = des_moms[0]
    ne_moms.label = '$N_{e}$'
    ne_moms.title = '$N$\n($cm^{-3}$)'
    ne_moms.plot_title = ''
    
    # Scalar pressure
    Pi_moms = dis_moms[2]
    Pe_moms = des_moms[2]
    
    pi_moms = (Pi_moms[:,0,0] + Pi_moms[:,1,1] + Pi_moms[:,2,2]) / 3.0
    pe_moms = (Pe_moms[:,0,0] + Pe_moms[:,1,1] + Pe_moms[:,2,2]) / 3.0
    
    # Entropy
    Si_dist = entropy(dis_dist, 'i')
    Se_dist = entropy(des_dist, 'e')
    
    Si_moms = maxwellian_entropy(ni_moms, pi_moms, 'i')
    Si_moms.label = '$S_{i,Max}$'
    Se_moms = maxwellian_entropy(ne_moms, pe_moms, 'e')
    Se_moms.label = '$S_{e,Max}$'
    
    # M-bar
    Mbar_dis = np.abs(Si_dist - Si_moms) / Si_moms
    Mbar_des = np.abs(Se_dist - Se_moms) / Se_moms
    
    Mbar_dis.x0 = Si_moms.x0
    Mbar_dis.label = '$\overline{M}_{i}$'
    Mbar_dis.title = '$\overline{M}$'
    Mbar_des.x0 = Se_moms.x0
    Mbar_des.label = '$\overline{M}_{e}$'
    Mbar_des.title = '$\overline{M}$'
    
    # T-par
    Ti_par = dis_moms[5]
    Te_par = des_moms[5]
    
    Ti_par.label = '$T_{i,||}$'
    Ti_par.title = '$T_{i}$'
    Te_par.label = '$T_{e,||}$'
    Te_par.title = '$T_{e}$'
    
    # T-perp
    Ti_perp = dis_moms[6]
    Te_perp = des_moms[6]
    
    Ti_perp.label = '$T_{i,perp}$'
    Ti_perp.title = '$T_{i}$\n(eV)'
    Te_perp.label = '$T_{e,perp}$'
    Te_perp.title = '$T_{e}$\n(eV)'
    
    # Anisotropy
    Ai = Ti_par / Ti_perp - 1
    Ae = Te_par / Te_perp - 1
    
    Ai.x0 = Ti_par.x0
    Ai.label = '$A_{i}$'
    Ai.title = 'A'    
    Ae.x0 = Te_par.x0
    Ae.label = '$A_{i}$'
    Ae.title = 'A'    
    
    fig, axes = metabase.MetaCache.plot(
                    [[Bx, By, Bz],
                     [ni_moms, ne_moms],
                     [Si_dist, Si_moms],
                     [Se_dist, Se_moms],
                     [Mbar_dis, Mbar_des],
                     [Ti_par, Ti_perp],
                     [Te_par, Te_perp],
                     [Ai, Ae]
                     ], nrows=8, ncols=1, figsize=(6, 7), legend=True
                    )
    
    fig.suptitle('Plasma Parameters for Kinetic and Boltzmann Entropy')
    plt.subplots_adjust(left=0.2, top=0.95)
    return fig, axes


def precondition(dist, E0=100, E_low=10, scpot=None):
    '''
    Before being sent to the integration routine, skymaps are preprocessed
    in the following manner:
      1. f(phi = 0) is repeated as f(phi = 360) to ensure that the periodic
         boundary condition is incorporated to the azimuthal integration.
      2. f(theta=0) = 0 and f(theta=180) = 0 data points are added to ensure
         the polar integration goes from 0 to 180.  The sin(theta)
         dependence of the polar integration force the integrand at
         theta = 0 and theta = 180 to zero regardless of the value of the
         phase space density
      3. f(U = 0) = 0 and f(U=1) =0 data points are added to ensure the
         integration goes from E->0 to E->infinity. V = 0 forces the
         integrand equal to zero regardless of the phase space density. 
    
    Parameters
    ----------
    dist : `metaarray.MetaArray`
        The velocity distribution function (s^3/cm^6) with azimuth, polar,
        and energy dependencies as attributes.
    E0 : float
        Energy value (eV) used when mapping energy bins from range [0,Emax]
        to [0, inf)
    E_low : float
        Energy value (eV) representing the low-energy cut-off
    '''
    # Integrate over phi
    phi = (np.deg2rad(np.append(dist.x1, dist.x1[0]+360.0))
           )[np.newaxis, :, np.newaxis, np.newaxis]
    f_out = np.append(dist, dist[:,0,np.newaxis,:,:], axis=1)
    
    # Integrate over theta
    theta = (np.deg2rad([0.0, *dist.x2, 180.0], dtype='float32')
             )[np.newaxis, :, np.newaxis]
    f_out = np.insert(f_out, 0, 0, axis=2)
    f_out = np.insert(f_out, f_out.shape[2], 0, axis=2)
    
    # Adjust for spacecraft potential
    if scpot is not None:
        pass
    
    # Integrate over Energy
    iGtElow= np.argmin((dist.x3[0, :] >= E_low) == False)
    U = dist.x3[:, iGtElow:] / (dist.x3[:, iGtElow:] + E0)
    U = np.insert(U, 0, 0, axis=1)
    U = np.insert(U, U.shape[1], 1, axis=1)
    f_out = f_out[:, :, :, iGtElow:]
    f_out = np.insert(f_out, 0, 0, axis=3)
    f_out = np.insert(f_out, f_out.shape[3], 0, axis=3)

    return f_out, phi, theta, U


def pressure(dist, species, N=None, T=None):
    kB = constants.k
    eV2K = constants.value('electron volt-kelvin relationship')
    if N is None:
        N = density(dist, species)
    if T is None:
        T = temperature(dist, species, N=N)
    
    P = 1e15 * N[:, np.newaxis, np.newaxis] * kB * eV2K * T
    
    P.x0 = dist.x0
    P.name = 'D{0}S pressure tensor'.format(species.upper())
    P.plot_title = ('Pressure calculated from d{}s velocity distribution '
                    'function.'.format(species))
    P.title = '$P$\n(nPa)'
    P.scale = 'linear'
    P.units = 'nPa'
    
    return P


def species_to_mass(species):
    
    if species == 'i':
        mass = constants.m_p
    elif species == 'e':
        mass = constants.m_e
    else:
        raise ValueError(('Unknown species {}. Select "i" or "e".'
                          .format(species))
                         )
    
    return mass


def temperature(dist, species, N=None, V=None):
    K2eV = constants.value('kelvin-electron volt relationship')
    eV2J = constants.eV
    kB = constants.k # J/k
    mass = species_to_mass(species)
    if N is None:
        N = density(dist, species)
    if V is None:
        V = velocity(dist, species, N=N)
    
    f_out, phi, theta, U = precondition(dist)
    
    # Integrate over phi
    Txx = int_trapezoid(phi, f_out * np.cos(phi)**2)
    Tyy = int_trapezoid(phi, f_out * np.sin(phi)**2)
    Tzz = int_trapezoid(phi, f_out)
    Txy = int_trapezoid(phi, f_out * np.cos(phi)  * np.sin(phi))
    Txz = int_trapezoid(phi, f_out * np.cos(phi))
    Tyz = int_trapezoid(phi, f_out * np.sin(phi))
    
    # Integrate over theta
    Txx = int_trapezoid(theta, Txx * np.sin(theta)**3)
    Tyy = int_trapezoid(theta, Tyy * np.sin(theta)**3)
    Tzz = int_trapezoid(theta, Tzz * np.sin(theta) * np.cos(theta)**2)
    Txy = int_trapezoid(theta, Txy * np.sin(theta)**3)
    Txz = int_trapezoid(theta, Txz * np.sin(theta)**2 * np.cos(theta))
    Tyz = int_trapezoid(theta, Tyz * np.sin(theta)**2 * np.cos(theta))
    
    # Integrate over energy
    E0 = 100
    Txx = np.divide(U**(3/2), (1-U)**(7/2)) * Txx
    Tyy = np.divide(U**(3/2), (1-U)**(7/2)) * Tyy
    Tzz = np.divide(U**(3/2), (1-U)**(7/2)) * Tzz
    Txy = np.divide(U**(3/2), (1-U)**(7/2)) * Txy
    Txz = np.divide(U**(3/2), (1-U)**(7/2)) * Txz
    Tyz = np.divide(U**(3/2), (1-U)**(7/2)) * Tyz
    Txx[:,-1] = 0
    Tyy[:,-1] = 0
    Tzz[:,-1] = 0
    Txy[:,-1] = 0
    Txz[:,-1] = 0
    Tyz[:,-1] = 0
    
    coeff = 1e6 * (2/mass)**(3/2) / (N * kB / K2eV) * (E0*eV2J)**(5/2)
    Txx = (coeff * int_trapezoid(U, Txx) - (1e6*mass/kB*K2eV*V[:,0]*V[:,0]))[:,np.newaxis]
    Tyy = (coeff * int_trapezoid(U, Tyy) - (1e6*mass/kB*K2eV*V[:,1]*V[:,1]))[:,np.newaxis]
    Tzz = (coeff * int_trapezoid(U, Tzz) - (1e6*mass/kB*K2eV*V[:,2]*V[:,2]))[:,np.newaxis]
    Txy = (coeff * int_trapezoid(U, Txy) - (1e6*mass/kB*K2eV*V[:,0]*V[:,1]))[:,np.newaxis]
    Txz = (coeff * int_trapezoid(U, Txz) - (1e6*mass/kB*K2eV*V[:,0]*V[:,2]))[:,np.newaxis]
    Tyz = (coeff * int_trapezoid(U, Tyz) - (1e6*mass/kB*K2eV*V[:,1]*V[:,2]))[:,np.newaxis]
    
    T = metaarray.MetaArray(np.concatenate((Txx, Txy, Txz,
                                            Txy, Tyy, Tyz,
                                            Txz, Tyz, Tzz),
                                           axis=1
                                           ).reshape(len(Txx), 3, 3)
                            )
    
    T.x0 = dist.x0
    T.name = 'D{0}S temperature tensor'.format(species.upper())
    T.plot_title = ('Temperature calculated from d{}s velocity distribution '
                    'function.'.format(species))
    T.title = '$T$\n(eV)'
    T.scale = 'linear'
    T.units = 'eV'
    
    return T


def velocity(dist, species, precondition=True, **kwargs):
    if precondition:
        vx, vy, vz = velocity_precondition(dist, species, **kwargs)
    else:
        vx, vy, vz = velocity_1(dist, species, **kwargs)
    
    Vx = metaarray.MetaArray(vx)
    Vy = metaarray.MetaArray(vy)
    Vz = metaarray.MetaArray(vz)
    
    Vy.x0 = dist.x0
    Vz.x0 = dist.x0
    
    bulkv = np.append(vx[:, np.newaxis], vy[:, np.newaxis], axis=1)
    bulkv = np.append(bulkv, vz[:, np.newaxis], axis=1)
    
    # X-component
    Vx.x0 = dist.x0
    Vx.name = 'D{0}S bulk velocity'.format(species.upper())
    Vx.plot_title = ('Bulk velocity calculated from d{}s velocity distribution '
                    'function.'.format(species))
    Vx.label = '$V_{' + species + 'x}$'
    Vx.title = '$V_{' + species + 'x}$\n(km/s)'
    Vx.scale = 'linear'
    Vx.units = 'km/s'
    
    # Y-component
    Vy.x0 = dist.x0
    Vy.name = 'D{0}S bulk velocity'.format(species.upper())
    Vy.plot_title = ('Bulk velocity calculated from d{}s velocity distribution '
                    'function.'.format(species))
    Vy.label = '$V_{' + species + 'y}$'
    Vy.title = '$V_{' + species + 'y}$\n(km/s)'
    Vy.scale = 'linear'
    Vy.units = 'km/s'
    
    # Z-component
    Vz.x0 = dist.x0
    Vz.name = 'D{0}S bulk velocity'.format(species.upper())
    Vz.plot_title = ('Bulk velocity calculated from d{}s velocity distribution '
                    'function.'.format(species))
    Vz.label = '$V_{' + species + 'z}$'
    Vz.title = '$V_{' + species + 'z}$\n(km/s)'
    Vz.scale = 'linear'
    Vz.units = 'km/s'

    return Vx, Vy, Vz
    

def velocity_1(dist, species, E0=100, E_low=10, scpot=None, N=None):
    
    kB = constants.k # J/K
    mass = species_to_mass(species)
    eV2J = constants.eV
    if N is None:
        N = density(dist, species)
    
    phi = (np.deg2rad(np.append(dist.x1, dist.x1[0]+360.0))
           )[np.newaxis, :, np.newaxis, np.newaxis]
    bulkv = np.append(dist, dist[:,0,np.newaxis,:,:], axis=1)
    
    # Integrate over theta
    theta = (np.deg2rad([0.0, *dist.x2, 180.0], dtype='float32')
             )[np.newaxis, :, np.newaxis]
    bulkv = np.insert(bulkv, 0, 0, axis=2)
    bulkv = np.insert(bulkv, bulkv.shape[2], 0, axis=2)
    
    bulkvx = int_trapezoid(phi, bulkv*np.cos(phi))
    bulkvy = int_trapezoid(phi, bulkv*np.sin(phi))
    bulkvz = int_trapezoid(phi, bulkv)

    bulkvx = int_trapezoid(theta, bulkvx * np.sin(theta)**2)
    bulkvy = int_trapezoid(theta, bulkvy * np.sin(theta)**2)
    bulkvz = int_trapezoid(theta, bulkvz * np.sin(theta) * np.cos(theta))

    v = np.sqrt(2 * eV2J * dist.x3 / mass) # m/s
    bulkvx = -1e3/N * int_trapezoid(v, bulkvx * v**3)
    bulkvy = -1e3/N * int_trapezoid(v, bulkvy * v**3)
    bulkvz = -1e3/N * int_trapezoid(v, bulkvz * v**3)
    
    return bulkvx, bulkvy, bulkvz
    

def velocity_precondition(dist, species, E0=100, E_low=10, scpot=None, N=None):
    
    kB = constants.k # J/K
    mass = species_to_mass(species)
    if N is None:
        N = density(dist, species)
    
    f_out, phi, theta, U = precondition(dist, E0=100, E_low=10)
    
    # Integrate over phi
    vx = int_trapezoid(phi, f_out*np.cos(phi))
    vy = int_trapezoid(phi, f_out*np.sin(phi))
    vz = int_trapezoid(phi, f_out)
    
    # Integrate over theta
    vx = int_trapezoid(theta, vx*np.sin(theta)**2)
    vy = int_trapezoid(theta, vy*np.sin(theta)**2)
    vz = int_trapezoid(theta, vz*np.cos(theta)*np.sin(theta))
    
    # Integrate over Energy
    E0 = 100
    yx = np.divide(U, (1-U)**3) * vx
    yy = np.divide(U, (1-U)**3) * vy
    yz = np.divide(U, (1-U)**3) * vz
    yx[:,-1] = 0
    yy[:,-1] = 0
    yz[:,-1] = 0
    
    vx = -1e3 * 2 * (constants.eV * E0 / mass)**2 / N * int_trapezoid(U, yx)
    vy = -1e3 * 2 * (constants.eV * E0 / mass)**2 / N * int_trapezoid(U, yy)
    vz = -1e3 * 2 * (constants.eV * E0 / mass)**2 / N * int_trapezoid(U, yz)
    
    return vx, vy, vz


if __name__ == '__main__':
    density()
    
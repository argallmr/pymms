import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.stats import binned_statistic
from pathlib import Path

# IGRF Coefficients
sample_data_path = Path(__file__).parent / 'swfo' / 'data'
igrf_coeff_file = sample_data_path / 'igrf13coeffs.txt'

class IGRF_Coeff():
    # See this link for:
    #   - https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
    #   - IGRF coefficients
    #   - Python library for IGRF reference field
    #   - Publication of IGRF-13
    
    def __init__(self):
        '''
        Initialize the object
        '''
        
        # Read in the table of IGRF coefficients
        self._coeffs = self.read_coeff()
    
    def closest_year(self, time):
        '''
        Return the index of the most recent past IGRF year.
        
        Parameters
        ----------
        time : `numpy.datetime64`
            Times at which to determine the IGRF coefficients
        '''
        
        # Years of input time and of coefficients
        years = pd.to_datetime(time).year.values
        igrf_years = self._coeffs.columns.values[:-1].astype(np.float32).astype(np.int32)

        # Locate the closest IGRF year: equivalent to $T_{t}$
        stat, edge, num = binned_statistic(years, years,
                                           statistic='count',
                                           bins=igrf_years,
                                           )

        return num
    
    def coeff(self, gh, m, n, time):
        '''
        Determine the IGRF coefficient at the given times.
        
        Parameters
        ----------
        gh : str
            The spherical harmonic coefficient, ('g', 'h')
        m : int
            The order of the spherical harmonic coefficient
        n : int
            The degree of the spherical harmonic coefficient
        time : `numpy.datetime64`
            Times at which to determine the IGRF coefficients
        
        Returns
        -------
        coeff_t : `numpy.array`
            The coefficient g_{n}^{m}(t) or h_{n}^{m}(t)
        '''
        if (time < np.datetime64('1900', 'Y')).any() | (time > np.datetime64('2026', 'Y')).any():
            raise ValueError('time must be in the range 1900-2025')
        
        # Coefficient of interest
        coeff = self.get_coeff(gh, m, n)
        
        # Most recent coefficient
        iyr = self.closest_year(time)
        coeff_T = coeff[:-1][iyr-1]
        
        # Linear change in coefficients over five years
        cdot_T = self.time_derivative(coeff, iyr)
        
        # Time difference from IGRF years
        dt = self.time_diff(time, iyr)

        # Linear interpolation of coefficients
        #   - Axes cannot be aligned during addition because the index has
        #     duplicate values
        #   - Add the values together instead of the DataFrames.
        coeff_t = coeff_T + dt * cdot_T
        coeff_t.index = pd.to_datetime(time).year
        
        return coeff_t
        
    def get_coeff(self, gh, m, n):
        '''
        Retrieve the values of a spherical harmonic coefficient from
        the look-up table.
        
        Parameters
        ----------
        gh : str
            The spherical harmonic coefficient, ('g', 'h')
        m : int
            The order of the spherical harmonic coefficient
        n : int
            The degree of the spherical harmonic coefficient
        
        Returns
        -------
        coeff_T : `numpy.array`
            The coefficient g_{n}^{m}(T) or h_{n}^{m}(T)
        '''
        return self._coeffs.loc[(gh, m, n), :]

    @staticmethod
    def read_coeff():
        '''
        Read the look-up table of spherical harmonic coefficients
        '''
        return pd.read_csv(igrf_coeff_file,
                           delim_whitespace=True,
                           header=3,
                           index_col=('g/h', 'n', 'm')
                           )

    @staticmethod
    def test(gh='g', n=1, m=0):
        '''
        Plot a coefficient and its interpolated values g_{n}^{m}(T)
        and g_{n}^{m}(t)
        
        Parameters
        ----------
        gh : str
            The spherical harmonic coefficient, ('g', 'h')
        m : int
            The order of the spherical harmonic coefficient
        n : int
            The degree of the spherical harmonic coefficient
        '''
        str_coeff = '${0:s}_{{{1:1d}}}^{{{2:1d}}}$'.format(gh, n, m)

        # Define a set of times at which to determine the coefficients
        t0 = np.datetime64('1900-01-01', 'Y')
        t1 = np.datetime64('2026-01-01', 'Y')
        times = np.arange(t0, t1, step=np.timedelta64(1, 'Y'))
    
        # Get the IGRF coefficients and the interpolated coefficients
        igrf = IGRF_Coeff()
        c = igrf.get_coeff(gh, n, m)[:-1]
        c_t = igrf.coeff(gh, n, m, times)
    
        # Plot the results
        from matplotlib import pyplot as plt
    
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.92)

        c.index = c.index.astype(np.float32).astype(np.int32)
        c_t.index = c_t.index.astype(np.float32).astype(np.int32)

        ax = axes[0,0]
        c_t.plot(marker='o', ax=ax, label=str_coeff[:-1]+'(t)$')
        c.plot(ax=ax, marker='x', label=str_coeff)
        ax.set_xlabel('Year')
        ax.set_ylabel(str_coeff + ' (nT)')
        ax.set_title('Interpolated IGRF Coefficients')
        ax.legend([str_coeff[:-1]+'(t)$', str_coeff[:-1]+'(T)$'])
    
        plt.show()
    
    def time_diff(self, time, iyr):
        '''
        Time difference from the IGRF look-up table times: dt = t - T
        
        Parameters
        ----------
        time : `numpy.datetime64`
            Times at which to determine the IGRF coefficients
        iyr : `numpy.array` int
            Index into the IGRF look-up table indicating the next
            earlier time (see `self.closest_year()`)
        
        Returns
        -------
        dt : `numpy.datetime64`
            Time difference from the IGRF times
        '''
        
        # IGRF years as datetimes
        igrf_yrs = np.array([c[0:4] 
                             for c in self._coeffs.columns[:-1]],
                            dtype='datetime64[Y]')
        
        # Compute fractional years
        dt = ((time - igrf_yrs[iyr-1]).astype('timedelta64[s]')
              / np.timedelta64(1, 'Y').astype('timedelta64[s]')
              )
        
        return dt
    
    @staticmethod
    def time_derivative(coeff, iyr):
        '''
        Time derivative of an IGRF coefficient
        
        Parameters
        ----------
        coeff : `numpy.array`
            Coefficient of which the derivative is computed
        iyr : `numpy.array` int
            Index into the IGRF look-up table indicating the next
            earlier time (see `self.closest_year()`)
        '''

        # Coefficient at nearest IGRF year
        c_T = coeff[:-1][iyr-1]

        # Coefficient at the next IGRF year, T+5
        #   - Note that the last column is already the time derivative
        c_Tplus5 = coeff[iyr]

        # Time derivative: 1/5 * [c(T+5) - c(T)]
        cdot = 0.2 * (c_Tplus5.values - c_T.values)

        # Create a DataFrame
        cdot = pd.Series(cdot, name='cdot', index=c_T.index)
        cdot_T = cdot.where((c_Tplus5.index != coeff.index[-1]),
                             other=c_Tplus5.values)

        return cdot_T


def date2mjd(date):
    '''
    Convert a date to Modified Julian Date -- the number WHOLE of days since
    1858-11-17T00:00:00Z.

    Calling Sequence:
      mjd = date2mjd(year, month, day)
      Calculate the Modified Julian Date from the numeric year, month and day.

      mjd = date2mjd (date)
      Calulate the Modified Julian Date from a date string, Nx10
      character array, or 1xN element cell array of strings, where N
      represents the number of dates formatted as 'yyyy-mm-dd'.

    Parameters
    ----------
    date : array of `numpy.datetime64`
        Date to be converted. Dates are converted to datetime64[D] during
        calculation
    
    Returns
    -------
    MJD : int
        Modified Julian Date

    References:
    - https://www.spenvis.oma.be/help/background/coortran/coortran.html
    - Hapgood, M. A. (1992). Space physics coordinate transformations:
      A user guide. Planetary and Space Science, 40 (5), 711?717.
      doi:http://dx.doi.org/10.1016/0032-0633 (92)90012-D
    - Hapgood, M. A. (1997). Corrigendum. Planetary and Space Science,
      45 (8), 1047 ?. doi:http://dx.doi.org/10.1016/S0032-0633 (97)80261-9
    '''
    
    # datetime64:
    #   Datetimes are always stored based on POSIX time (though having a TAI
    #   mode which allows for accounting of leap-seconds is proposed), with an
    #   epoch of 1970-01-01T00:00Z.
    #
    # MJD:
    #   The MJD gives the number of days since midnight on November 17, 1858.
    #   This date corresponds to 2400000.5 days after day 0 of the Julian
    #   calendar.
    
    mjd0 = np.datetime64('1858-11-17T00:00:00', 's')

    return (date - mjd0) / np.timedelta64(86400, 's')


def date2mjd2000(time):
    
    jul_2000 = date2mjd(np.datetime64('2000-01-01T12:00:00', 's'))
    mjd = date2mjd(time.astype('datetime64[D]'))
    return mjd - jul_2000


def date2ssm(time):
    '''
    Convert time to seconds since midnight. Note that if `time` contains times
    from different dates, that will not be reflected in the results.
    
    Parameters
    ----------
    time : `numpy.datetime64`
        Time to be converted
    
    Returns
    -------
    ssm : int, `numpy.ndarray`
        Fractional seconds elapsed since the previous midnight.
    '''
    return ((time - time.astype('datetime64[D]')).astype('timedelta64[ns]')
            ).astype(int) / 1e9


def date2ssny(time):
    '''
    Convert time to seconds since new year. Note that if `time` contains times
    from different years, that will not be reflected in the results.
    
    Parameters
    ----------
    time : `numpy.datetime64`
        Time to be converted
    
    Returns
    -------
    ssny : int, `numpy.ndarray`
        Fractional seconds elapsed since new year's eve.
    '''
    days = (time.astype('datetime64[D]') - time.astype('datetime64[Y]')
            ).astype('timedelta[s]').astype(int)
    ssm = date2ssm(time)
    return days + ssm


def date2sse(time, epoch=None, unit='s'):
    '''
    Convert time to seconds since new year. Note that if `time` contains times
    from different years, that will not be reflected in the results.
    
    Parameters
    ----------
    time : `numpy.datetime64`
        Time to be converted
    
    Returns
    -------
    ssny : int, `numpy.ndarray`
        Fractional seconds elapsed since new year's eve.
    '''
    if (epoch is None) | (epoch in ('new year', 'ny')):
        epoch = time.astype('datetime64[Y]')
    elif epoch == 'midnight':
        epoch = time.astype('datetime64[D]')
    elif epoch == 'first':
        epoch = time[0].astype('datetime64[D]')

    t_delta = (time - epoch).astype('timedelta64[ns]')
    
    if unit == 'D':
        divisor = 1e9 * 86400
    elif unit == 'h':
        divisor = 1e9 * 3600
    elif units == 'm':
        divisor = 1e9 * 60
    elif unit == 's':
        divisor = 1e9
    elif unit == 'ms':
        divisor = 1e6
    elif unit == 'us':
        divisor = 1e3
    elif unit == 'ns':
        divisor = 1
    else:
        raise ValueError('Unit not recognized. Must be denomination <= D')

    return t_delta.astype(float) / divisor


def date2juldays(time):
    '''
    Convert Gregorian date to Julian days.
    
    Julian days are calculated from noon on 1-Jan. 1900.
    
    Parameters
    ----------
    time : array of datetime64
    	Gregorian datetimes to be converted to Julian days
    
    Returns
    -------
    juldays : ndarray
    	Julian days elapsed since noon on 1-Jan. 1900 
    '''
    # Make sure the date is within range.
    if ((time < np.datetime64('1901', 'Y')).any()
        | (time > np.datetime64('2099', 'Y')).any()
        ):
        
        raise ValueError('Year must be between 1901 and 2099')

    # We need to convert to Julian centuries since 1900-01-01T12:00:00
    #   - Start by calculating the number of days
    
    # Whole years since 1900
    years = time.astype('datetime64[Y]') - np.datetime64('1900', 'Y')
    
    # Number of leap days since 1900 (4 years per leap day)
    #   - Julian Year: Exactly 365.25 days of 86,400 SI seconds each.
    try:
        leap_days = np.asarray(((time.astype('datetime64[Y]')
                                 - np.datetime64('1901', 'Y')
                                 ) / np.timedelta64(4, 'Y')
                                ).astype(int), 'timedelta64[D]'
                               )
    except Exception:
        import pdb
        pdb.set_trace()
    
    # Number of days into the current year
    year_day = (time.astype('datetime64[D]') - time.astype('datetime64[Y]')
                + np.timedelta64(1, 'D'))
    
    # Fraction of the current day that has elapsed
    frac_day = ((time - time.astype('datetime64[D]'))
                / np.timedelta64(1, 'D').astype('timedelta64[s]'))

    # Number of days elapsed since 1900-01-01T12:00:00
    #   - Note that we subtract 0.5 because the epoch starts at noon 
    Julian_days = ((365 * years.astype(int) + leap_days + year_day
                   ).astype(float) - 0.5) + frac_day
    
    return Julian_days


def earth_obliquity(T0):
    '''
    Axial tilt (obliquity) is the angle between an object's rotational axis
    and its orbital axis; equivalently, the angle between its equatorial
    plane and orbital plane.

    Parameters
    ----------
    T0 : `numpy.ndarray`
        Time in Julian centuries calculated from 12:00:00 UTC
        on 2000-01-01 (known as Epoch 2000) to the previous
        midnight. It is computed as:
        T0 = (MJD - 51544.5) / 36525.0

    Returns
    -------
    obliquity : `numpy.ndarray`
        Obliquity of Earth's ecliptic orbit (degrees).
    '''
    # 23.439 is the dipole tilt angle in degrees at Earth's position on
    # Jan. 1, 2000. 0.013 is the fractional number of degrees that this
    # angle changes per day: (1 - 360 degrees / 365 days/year)
    return 23.439 - 0.013 * T0


def dipole_igrf_coeffs(time):
    '''
    Compute the IGRF coefficients required to determine the dipole
    tilt axis and angle.
        
    Parameters
    ----------
    time : `numpy.datetime64`
        Times at which to determine the IGRF coefficients
        
    Returns
    -------
    g10 : `numpy.float`
        g10 IGRF coefficient
    g11 : `numpy.float`
        g11 IGRF coefficient
    h11 : `numpy.float`
        h11 IGRF coefficient
    '''
    igrf = IGRF_Coeff()
    g10 = igrf.coeff('g', 1, 0, time)
    g11 = igrf.coeff('g', 1, 1, time)
    h11 = igrf.coeff('h', 1, 1, time)
    
    return g10, g11, h11


def dipole_latlon(time):
    '''
    Compute the latitude and longitude of the dipole axis in GEO coordinates.
        
    Parameters
    ----------
    time : `numpy.datetime64`
        Times at which to determine the coordinates of the dipole axis
        
    Returns
    -------
    lat : `numpy.float`
        Latitude of Earth's dipole axis in GEO coordinates in degrees
    lon : `numpy.float`
        Longitude of Earth's dipole axis in GEO coordinates in degrees
    '''
    g10, g11, h11 = dipole_igrf_coeffs(time)
    lat, lon = dipole_igrf2latlon(g10, g11, h11)
    return lat, lon


def dipole_unit_vector(time):
    '''
    Compute the (x, y, z) GEO coordinates of Earth's dipole axis unit vector.
        
    Parameters
    ----------
    time : `numpy.datetime64`
        Times at which to determine the coordinates of the dipole axis
        
    Returns
    -------
    Q_geo : `numpy.float`
        Earth's dipole axis as a unit vector in GEO coordinates
    '''
    lat, lon = dipole_latlon(time)
    Q_geo = dipole_latlon2vector(lat, lon)
    return Q_geo


def dipole_igrf2latlon(g10, g11, h11):
    '''
    Convert IGRF coefficients that define Earth's magnetic dipole axis
    to latitude and longitude values in GEO coordinates.
        
    Parameters
    ----------
    g10 : `numpy.float`
        g10 IGRF coefficient
    g11 : `numpy.float`
        g11 IGRF coefficient
    h11 : `numpy.float`
        h11 IGRF coefficient
        
    Returns
    -------
    lat : `numpy.float`
        Latitude of Earth's dipole axis in GEO coordinates in degrees
    lon : `numpy.float`
        Longitude of Earth's dipole axis in GEO coordinates in degrees
    '''
    
    geo_lon = np.rad2deg(np.arctan2(h11, g11))
    geo_lat = 90.0 - np.rad2deg(np.arctan((g11*np.cos(geo_lon)
                                           + h11*np.sin(geo_lon)
                                           ) / g10))
    return geo_lat, geo_lon


def dipole_latlon2vector(lat, lon):
    '''
    Convert the latitude and longitude of Earth's magnetic dipole axis
    to a unit vector in GEO coordinates.
        
    Parameters
    ----------
    lat : `numpy.float`
        Latitude of Earth's dipole axis in GEO coordinates in degrees
    lon : `numpy.float`
        Longitude of Earth's dipole axis in GEO coordinates in degrees
        
    Returns
    -------
    Q_geo : `numpy.float`
        Earth's dipole axis as a unit vector in GEO coordinates
    '''
    dlat = np.deg2rad(lat)
    dlon = np.deg2rad(lon)
    return np.column_stack([np.cos(dlat)*np.cos(dlon),
                            np.cos(dlat)*np.sin(dlon),
                            np.sin(dlon)])


def dipole_time2latlon(time):
    '''
    Calculate the latitude and longitude of Earth's magnetic dipole axis
    in GEO coordinates.
        
    Parameters
    ----------
    time : `numpy.datetime64`
        Times at which to determine the coordinates of the dipole axis
        
    Returns
    -------
    lat : `numpy.float`
        Latitude of Earth's dipole axis in GEO coordinates in degrees
    lon : `numpy.float`
        Longitude of Earth's dipole axis in GEO coordinates in degrees
    '''
    mjd = time2mjd(time)
    lat = 78.8 + 4.283e-2 * (mjd - 46066) / 365.25
    lon = 289.1 - 1.413e-2 * (mjd - 46066) / 365.25
    
    return lat, lon


def dipole_axis(lat=None, lon=None,
                g10=None, g11=None, h11=None, time=None):
    '''
    Determine Earth's magnetic dipole axis as a unit vector in
    GEO coordinates.
        
    Parameters
    ----------
    lat : `numpy.float`
        Latitude of Earth's dipole axis in GEO coordinates
    lon : `numpy.float`
        Longitude of Earth's dipole axis in GEO coordinates
    g10 : `numpy.float`
        g10 IGRF coefficient
    g11 : `numpy.float`
        g11 IGRF coefficient
    h11 : `numpy.float`
        h11 IGRF coefficient
    time : `numpy.datetime64`
        Times at which to determine the coordinates of the dipole axis
        
    Returns
    -------
    Q_geo : `numpy.float`
        Earth's dipole axis as a unit vector in GEO coordinates
    '''
    
    if time is not None:
        Q_geo = dipole_unit_vector(time)
    elif (g10 is not None) & (g11 is not None) & (h11 is not None):
        lat, lon = dipole_igrf2latlon(g10, g11, h11)
    
    if (lat is not None) & (lon is not None):
        Q_geo = dipole_latlon2vector(lat, lon)
    
    return Q_geo


def dipole_inclination(time):
    '''
    The angle between the GSE Z axis and projection of the magnetic
    dipole axis on the GSE YZ plane (i.e. the GSM Z axis) measured
    positive for rotations towards the GSE Y axis.
    
    Parameters
    ----------
    time : `numpy.datetime64`
        Times at which to calculate the dipole inclination angle.
    
    Returns
    -------
    psi : float
        Inclination angle between z-GSM and z-GSe
    '''
    
    Q_geo = dipole_axis(time=time)
    
    # Rotate GEO coordinates to GSE coordinates
    r_geo2gse = geo2gse(time)
    axis = r_geo2gse.apply(Q_geo)
    
    # Compute the dipole tilt angle
    psi = np.arctan2(axis[:,1], axis[:,2])
    
    return psi


def dipole_tilt_angle(time):
    '''
    Dipole tilt angle, i.e. the angle between the GSM Z axis and the
    dipole axis. It is positive for the North dipole pole sunward of GSM Z.
    
    Parameters
    ----------
    time : `numpy.datetime64`
        Times at which to calculate the dipole tilt angle.
    
    Returns
    -------
    mu : float
        Tilt angle between z-GSM and the dipole axis
    '''
    
    Q_geo = dipole_axis(time=time)
    
    # Rotate GEO coordinates to GSE coordinates
    r_geo2gse = geo2gse(time)
    Q_gse = r_geo2gse.apply(Q_geo)
    
    # Compute the dipole tilt angle
    mu = np.arctan(Q_gse[:,0] / np.sqrt(Q_gse[:,1]**2 + Q_gse[:,2]**2))
    
    return mu


def gei2dsc(time, ra, dec):
    '''
    Return the transformation to the despun spacecraft frame (SCS) from
    Geocentric Equatorial Inertial system (GEI) at the given time, with RA
    and dec (in degrees) of the spin vector.

    Parameters
    ----------
    YEAR : in, required, type = double
    MONTH : in, required, type = double
    DAY : in, required, type = double
    SECS : in, required, type = double
        Seconds into `DAY`.
    RA : in, required, type = double
        Right-ascention of the spacecraft (degrees).
    DEC : in, required, type = double
        Declination of the spacecraft (degrees).

    Returns
    -------
    SCS2GSE : out, optional, type = float
        Transformation matrix to rotate SCS to GSE.

    References
    ----------
    - https://www.spenvis.oma.be/help/background/coortran/coortran.html
    - Hapgood, M. A. (1992). Space physics coordinate transformations:
        A user guide. Planetary and Space Science, 40 (5), 711?717.
        doi:http://dx.doi.org/10.1016/0032-0633 (92)90012-D
    - Hapgood, M. A. (1997). Corrigendum. Planetary and Space Science,
        45 (8), 1047 ?. doi:http://dx.doi.org/10.1016/S0032-0633 (97)80261-9
    '''

    # Location of the sun
    SUN = sun_position(time)  # in what coords? normalized?

    # RA and DEC form a spherical coordinate system.
    # - RA  = number of hours past the vernal equinox (location on the
    #         celestial equator of sunrise on the first day of spring).
    # - DEC = degrees above or below the equator
    dec_rad = np.deg2rad(dec)
    ra_rad = np.deg2rad(ra)

    # [x y z] components of the unit vector pointing in the direction of
    # the spin axis.
    # - The spin axis points to a location on the suface of the celestial sphere.
    # - RA and dec are the spherical coordinates of that location,
    #   with the center of the earth as the origin.
    # - Transforming GEI to SCS transforms [0 0 1] to [x y z] = OMEGA
    # - Already normalized: spherical to cartesian with r = 1.
    scsz  = np.column_stack((np.cos(ra_rad) * np.cos(dec_rad),
                             np.sin(ra_rad) * np.cos(dec_rad),
                             np.sin(dec_rad)
                             ))

    # Form the X- and Y-vectors
    # - X must point in the direction of the sun.
    # - To ensure this, Y' = Z' x Sun
    # - X' = Y' x Z'
    scsy = np.cross(scsz, SUN, axis=1)
    scsy /= np.linalg.norm(scsy, axis=1)[:, np.newaxis]
    scsx = np.cross(scsy, scsz, axis=1)
    
    # scsy = mrvector_cross(scsz, SUN)
    # scsy = mrvector_normalize(scsy)
    # scsx = mrvector_cross(scsy, scsz)

    # Transformation from GEI to SCS.
    GEI2DSC = np.zeros((len(ra), 3, 3))
    GEI2DSC[:,0,:] = scsx
    GEI2DSC[:,1,:] = scsy
    GEI2DSC[:,2,:] = scsz

    return R.from_matrix(GEI2DSC)


def gei2geo(time):
    
    T0 = nJulCenturies(date2mjd2000(time))
    UT = date2ssm(time)
    
    theta = 100.461 + 36000.770*T0 + 15.04107*UT
    
    # scipy rotates the vector. We want to rotate the coordinate system.
    return R.from_euler('Z', theta, degrees=True).inv()


def geo2gse(time):
    T1 = gei2geo(time).inv()
    T2 = gei2gse(time)
    return T2 * T1


def geo2mag(time):
    lat, lon = dipole_latlon(time)
    
    # scipy rotates the vector. We want to rotate the coordinate system.
    return R.from_euler('Y', latitude-90).inv() * R.from_euler('Z', longitude).inv()


def gei2gse(time):
    '''
    Produce a rotation matrix from GEI to GSE.

    Parameters
    ----------
    MJD : int
        Modified Julian Date.
    UTC : `numpy.ndarray`
        UTC in decimal hours since midnight.

    Returns
    -------
    T3 : out, required, type = double
        Totation matrix from GEI to GSE.

    References:
      See Hapgood Rotations Glossary.txt.
      - https://www.spenvis.oma.be/help/background/coortran/coortran.html
      - Hapgood, M. A. (1992). Space physics coordinate transformations:
          A user guide. Planetary and Space Science, 40 (5), 711?717.
          doi:http://dx.doi.org/10.1016/0032-0633 (92)90012-D
      - Hapgood, M. A. (1997). Corrigendum. Planetary and Space Science,
          45 (8), 1047 ?. doi:http://dx.doi.org/10.1016/S0032-0633 (97)80261-9
    '''

    # Number of julian centuries since Epoch 2000
    jul_cent = nJulCenturies(date2mjd2000(time))
    UTC = date2ssm(time) / 3600

    # Axial tilt
    obliq = earth_obliquity(jul_cent)
    eLon  = sun_ecliptic_longitude(jul_cent, UTC)

    #
    # The transformation from GEI to GSE, then is
    #   - T2 = <eLon, Z> <obliq, X>
    #   - A pure rotation about X by angle obliq
    #   - A pure rotation about Z by angle eLon
    #
    
    # Scipy rotates the vector. We want to rotate the coordinate system.
    T21 = R.from_euler('X', obliq, degrees=True).inv()
    T22 = R.from_euler('Z', eLon, degrees=True).inv()
    T2 = T22 * T21
    
    return T2


def gse2gsm(time):
    
    phi = dipole_inclination(time)
    
    # scipy rotates the vector. We want to rotate the coordinate system
    T3 = R.from_euler('X', -phi, degrees=True).inv()
    
    return T3


def gsm2sm(time):

    # Dipole tilt angle
    mu = dipole_tilt_angle(time)
    
    #  scipy rotates the vector. We want to rotate the coordinate system
    return R.from_euler('Y', -mu).inv()


def mjd2epoch2000(mjd):
    '''
    Convert Modified Julian Date (MJD) to Epoch 2000
       MJD: Number of days since midnight on November 17, 1858
       Epoch2000: Number of days since noon on January 1, 2000

    Parameters
    ----------
    mjd : `numpy.ndarray`
        Modified Julian dates

    Returns
    -------
    epoch2000 : `numpy.ndarray`
        Epoch 2000 times
    '''
    return mjd - 51544.5


def nJulCenturies(nDays):
    '''
    Convert number of days to Julian Centuries. there are exactly 36525 days
    in a Julian Century

    Parameters
    ----------
    nDays : `numpy.ndarray`
        Fractional number of days

    Returns
    -------
    jul_centuries : `numpy.ndarray`
        Number of Julian centuries
    '''
    return nDays / 36525.0


def sun_ecliptic_longitude(T0, UTC):
    ''''
    Determine the ecliptic longitude of the sun

    Note:
    Strictly speaking, TDT (Terrestrial Dynamical Time) should be used
    here in place of UTC, but the difference of about a minute gives a
    difference of about 0.0007∞ in lambdaSun.

    Calling Sequence:
    eLon = sun_ecliptic_longitude (T0, UTC)
    Compute the sun's ecliptic longitude (degrees) given the number
    of julian centuries (T0) from 12:00 UTC 01-Jan-2000 until
    00:00 UTC on the day of interest, and Universal Time (UTC) in
    fractional number of hours.
    
    Parameters
    ----------
    T0 : `numpy.ndarray`
        Time in Julian centuries calculated from 12:00:00 UTC
        on 1 Jan 2000 (known as Epoch 2000) to the previous
        midnight. It is computed as:
        T0 = (MJD - 51544.5) / 36525.0
    
    UTC : `numpy.ndarray`
        UTC decimal hours since midnight
    
    Returns
    -------
    eLon : `numpy.ndarray`
        Mean anomaly of the sun, in degrees
    
    References
    ----------
    See Hapgood Rotations Glossary.txt.
    - https://www.spenvis.oma.be/help/background/coortran/coortran.html
    - Hapgood, M. A. (1992). Space physics coordinate transformations:
      A user guide. Planetary and Space Science, 40 (5), 711?717.
      doi:http://dx.doi.org/10.1016/0032-0633 (92)90012-D
    - Hapgood, M. A. (1997). Corrigendum. Planetary and Space Science,
      45 (8), 1047 ?. doi:http://dx.doi.org/10.1016/S0032-0633 (97)80261-9
    '''
    # Sun's Mean anomaly
    ma = np.deg2rad(sun_mean_anomaly(T0, UTC))

    # Mean longitude (degrees)
    mLon = sun_mean_longitude(T0, UTC)

    # Ecliptic Longitude
    #   - Force to the range [0, 360)
    eLon = (mLon
            + (1.915 - 0.0048 * T0) * np.sin(ma)
            + 0.020 * np.sin(2.0 * ma)
            ) % 360.0

    return eLon


def sun_mean_anomaly(T0, UTC):
    '''
    Compute the sun's mean anomaly.

    Note:
    Strictly speaking, TDT (Terrestrial Dynamical Time) should be used
    here in place of UTC, but the difference of about a minute gives a
    difference of about 0.0007∞ in lambdaSun.

    Calling Sequence:
    ma = sun_mean_anomaly(T0, UTC)
    Compute the sun's mean anomaly (degrees) given the number
    of Julian centuries (T0) from 2000-01-01T12:00:00Z until
    00:00 UTC on the day of interest, and Universal Time (UTC) in
    decimal hours.
    
    Parameters
    ----------
    T0 : `numpy.ndarray`
        Time in Julian centuries calculated from 2000-01-01T12:00:00Z
        (known as Epoch 2000) to the previous midnight. It is computed as
            T0 = (MJD - 51544.5) / 36525.0
    UTC : `numpy.ndarray`
        UTC decimal hours since midnight
    
    Returns
    -------
    mean_anomaly : `numpy.ndarray`
        Ecliptic longitude of the sun, in degrees.
    
    References
    ----------
      See Hapgood Rotations Glossary.txt.
      - https://www.spenvis.oma.be/help/background/coortran/coortran.html
      - Hapgood, M. A. (1992). Space physics coordinate transformations:
        A user guide. Planetary and Space Science, 40 (5), 711?717.
        doi:http://dx.doi.org/10.1016/0032-0633 (92)90012-D
      - Hapgood, M. A. (1997). Corrigendum. Planetary and Space Science,
        45 (8), 1047 ?. doi:http://dx.doi.org/10.1016/S0032-0633 (97)80261-9
    '''
    # Sun's Mean anomaly
    #   - Force to the range [0, 360)
    return (357.528 + 35999.050 * T0 + 0.04107 * UTC) % 360.0


def sun_mean_longitude(T0, UTC):
    '''
    Compute the sun's mean longiude.

    Note:
    Strictly speaking, TDT (Terrestrial Dynamical Time) should be used
    here in place of UTC, but the difference of about a minute gives a
    difference of about 0.0007∞ in lambdaSun.

    Calling Sequence:
    lambda = sun_mean_anomaly(T0, UTC)
    Compute the sun's mean longitude (degrees) given the number
    of julian centuries (T0) from 2000-01-01T12:00:00Z until
    00:00 UTC on the day of interest, and Universal Time (UTC) in
    decimal hours.
    
    Parameters
    ----------
      T0:     in, required, type = double
              Time in Julian centuries calculated from 2000-01-01T12:00:00Z
              (known as Epoch 2000) to the previous midnight. It is computed as
                T0 = (MJD - 51544.5) / 36525.0
      UTC:    in, required, type = double
        UTC decimal hours since midnight
    
    
    Returns
    -------
    LAMBDA: out, required, type = double
        Mean longitude of the sun, in degrees.
    
    References
    ----------
    - https://www.spenvis.oma.be/help/background/coortran/coortran.html
    - Hapgood, M. A. (1992). Space physics coordinate transformations:
      A user guide. Planetary and Space Science, 40 (5), 711?717.
      doi:http://dx.doi.org/10.1016/0032-0633 (92)90012-D
    - Hapgood, M. A. (1997). Corrigendum. Planetary and Space Science,
      45 (8), 1047 ?. doi:http://dx.doi.org/10.1016/S0032-0633 (97)80261-9
    '''
    # Sun's Mean Longitude
    #   - Force to the range [0, 360)
    return (280.460 + 36000.772 * T0 + 0.04107 * UTC) % 360

def sun_position(time):
    '''
    Determine the direction of the sun in GEI.

    Program to caluclate sidereal time and position of the sun. It is good
    for years 1901 through 2099 due to leap-year limitations. Its accuracy
    is 0.006 degrees.

    Direction of the sun in cartesian coordinates:
        X = cos(SRASN) cos(SDEC)
        Y = sin(SRASN) cos(SDEC)
        Z = sin(SDEC)

    Parameters
    ----------
        time : `numpy.datetime64`
            Year.

    Returns
    -------
        S : out, optional, type=float
            Sidereal time and position of the sun.

    References
    ----------
    C.T. Russel, Geophysical Coordinate Transformations.
    http://www-ssc.igpp.ucla.edu/personnel/russell/papers/gct1.html/#appendix2
    
    The above link appears to be broken (2021-11-11). Here is another link.
    See Appendix 2.
    http://jsoc.stanford.edu/~jsoc/keywords/Chris_Russel/Geophysical%20Coordinate%20Transformations.htm#appendix2

    See the following reference, page C5, for a brief description of the
    algorithm (with updated coefficients?).

        (US), N. A. O. (Ed.). (2013). The Astronomical Almanac for the Year
            2014. U.S. Government Printing Office. Retrieved from
            http://books.google.com/books?id=2E0jXu4ZSQAC

    This reference provides an algroithm that reduces the error and extends
    the valid time range (i.e. not this algorithm).

        Reda, I.; Andreas, A. (2003). Solar Position Algorithm for Solar
            Radiation Applications. 55 pp.; NREL Report No. TP-560-34302,
            Revised January 2008. http://www.nrel.gov/docs/fy08osti/34302.pdf
    '''
    # Seconds elapsed into day
    frac_day = (time - time.astype('datetime64[D]')) / np.timedelta64(1, 's')
    
    # Number of days elapsed since 1900-01-01T12:00:00
    Julian_days = date2juldays(time)

    # Constants
    # RAD = 57.29578         # 180/pi

    # Convert seconds to days.
    # FDAY = SECS/86400

    # Number of days since noon on 1 Jan 1900.
    # DDJ = 365 .* (IYR-1900) + fix((IYR-1901) / 4) + IDAY - 0.5 
    # DJ = DDJ .* ones(1, length(SECS)) + FDAY  

    # Convert to Julian centuries from 1900
    #   - Julian Year:    Exactly 365.25 days of 86,400 SI seconds each.
    #   - Julian Century: 36,525 days
    T = nJulCenturies(Julian_days)
    # T = DJ / 36525

    # Degrees per day
    #   - It takes 365.2422 days to complete a revolution about the sun.
    #   - There are 360 degrees in a circle.
    #  => 360.0 / 365.2422 = 0.9856 degrees/day

    # Keep degrees between 0 and 360
    #   mod(..., 360) will force answer to be in range [0, 360).


    # Mean longitude of the sun
    VL = (279.696678 + 0.9856473354 * Julian_days) % 360.0

    # Greenwhich sidereal time.
    GST = (279.690983 + 0.9856473354 * Julian_days
           + 360 * frac_day + 180.) % 360.0

    # Mean anomaly
    G = np.deg2rad((358.475845 + 0.985600267 * Julian_days) % 360)

    # Ecliptic longitude
    SLONG = (VL + (1.91946 - 0.004789 * T) * np.sin(G)
             + 0.020094 * np.sin(2 * G))

    # Obliquity (Axial tilt)
    OBLIQ = np.deg2rad(23.45229 - 0.0130125 * T)


    SLP = np.deg2rad(SLONG - 0.005686)
    SIND = np.sin(OBLIQ) * np.sin(SLP)
    COSD = np.sqrt(1 - SIND**2)

    # Solar declination
    SDEC = np.rad2deg(np.arctan(SIND / COSD))

    # Solar right ascension
    SRASN  = 180 - np.rad2deg(np.arctan2(SIND / (np.tan(OBLIQ) * COSD),
                              -np.cos(SLP) / COSD))

    # Equatorial rectangular coordinates of the sun
    S = np.column_stack((np.cos(np.deg2rad(SRASN)) * COSD,
                         np.sin(np.deg2rad(SRASN)) * COSD,
                         SIND))

    return S

""" Module to convert between TAI and UTC time scales.

    utc2tai(datetime.datetime.now())
        returns a datetime.datetime object representing TAI from a datetime.datetime
        object at UTC.
    tai2utc(datetime(1992, 6, 2, 8, 7, 9))
        returns a datetime.datetime object representing UTC from a datetime.datetime
        object at TAI.
    taisec2utc(1400000000.0)
        returns a datetime.datetime object representing TAI from a real value
        representing seconds since the TAI epoch
    

    Originally from:
        https://pypi.python.org/pypi/tai64n
        modifications:
            add 2012+ leapseconds
            add taisec2utc, utc2taisec
            fix tai2utc for times near a leapsecond
            remove hex string decoding
        @author: Kim Kokkonen
"""


from datetime import timedelta, datetime
from operator import itemgetter
import time
import pytz

def utc_datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None):
    '''Returns a datetime with a UTC timezone'''
    dt = datetime(year, month, day, hour, minute, second, microsecond, tzinfo=tzinfo)
    dt = to_utc_tz(dt)
    return dt


def to_utc_tz(dt):
    '''Return a datetime properly converted to the UTC timezone.'''
    if str(dt.tzinfo) == 'UTC':
        return dt

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    dt = dt.astimezone(pytz.UTC)
    return dt


# offset in seconds between unix and TAI epochs
unix_epoch = utc_datetime(1970, 1, 1)
tai_unix_offset = (unix_epoch - utc_datetime(1958, 1, 1)).total_seconds()


def __conversion_table():
    """ returns [datetime, value] ordered reverse by date
        where value == seconds between TAI and UTC

    Example:
    >>> __conversion_table()[0]
    (datetime.datetime(1972, 1, 1, 0, 0), 10.0)
    """
    # update this table as new values become known
    # source: ftp://maia.usno.navy.mil/ser7/tai-utc.dat
    conversion_table = [(utc_datetime(1972, 0o1, 1), 10.0),
                        (utc_datetime(1972,0o7, 1), 11.0),
                        (utc_datetime(1973,0o1, 1), 12.0),
                        (utc_datetime(1974,0o1, 1), 13.0),
                        (utc_datetime(1975,0o1, 1), 14.0),
                        (utc_datetime(1976,0o1, 1), 15.0),
                        (utc_datetime(1977,0o1, 1), 16.0),
                        (utc_datetime(1978,0o1, 1), 17.0),
                        (utc_datetime(1979,0o1, 1), 18.0),
                        (utc_datetime(1980,0o1, 1), 19.0),
                        (utc_datetime(1981,0o7, 1), 20.0),
                        (utc_datetime(1982,0o7, 1), 21.0),
                        (utc_datetime(1983,0o7, 1), 22.0),
                        (utc_datetime(1985,0o7, 1), 23.0),
                        (utc_datetime(1988,0o1, 1), 24.0),
                        (utc_datetime(1990,0o1, 1), 25.0),
                        (utc_datetime(1991,0o1, 1), 26.0),
                        (utc_datetime(1992,0o7, 1), 27.0),
                        (utc_datetime(1993,0o7, 1), 28.0),
                        (utc_datetime(1994,0o7, 1), 29.0),
                        (utc_datetime(1996,0o1, 1), 30.0),
                        (utc_datetime(1997,0o7, 1), 31.0),
                        (utc_datetime(1999,0o1, 1), 32.0),
                        (utc_datetime(2006,0o1, 1), 33.0),
                        (utc_datetime(2009,0o1, 1), 34.0),
                        (utc_datetime(2012,0o7, 1), 35.0),
                        (utc_datetime(2015,0o7, 1), 36.0),
                        (utc_datetime(2017,0o1, 1), 37.0),
                        # add new values here
                        ]
    conversion_table.sort(key=itemgetter(0), reverse=True)
    return conversion_table


def __tai_seconds(date, table=None):
    """ returns seconds of TAI-offset from UTC at date given.
        Works only on dates later than0o1.01.1972.

    Example:
        >>> __tai_seconds(utc_datetime(1992, 6, 2, 8, 7, 9))
        26.0
        >>> __tai_seconds(utc_datetime(1971, 6, 2, 8, 7, 9))
        False
    """

    # Avoids risk of conversion table being changed
    if table is None:
        table = __conversion_table()

    for x in table:
        if date > x[0]:
            return x[1]
    return False


def tai2utc(date):
    """ converts datetime.datetime TAI to datetime.datetime UTC.
        Works only on dates later than0o1.01.1972.

    Example
        >>> tai2utc(utc_datetime(1992, 6, 2, 8, 7, 9))
        datetime.datetime(1992, 6, 2, 8, 6, 43)
    """
    # leapseconds at the tai time
    seconds = __tai_seconds(date)
    # first approximation to utc datetime
    dateutc = date - timedelta(0, seconds)
    # leap seconds at corresponding utc datetime
    secondsu = __tai_seconds(dateutc)

    if seconds > secondsu:
        # utc and tai on different sides of a leapsecond
        if date.second < secondsu:
            # tai time before the new leapsecond is applied
            return seconds and (date - timedelta(0, secondsu))
        # new leapsecond is applied
        # datetime can't represent 23:59:60 so return 23:59:59 again
        return secondsu and dateutc
    # no leapsecond shift involved
    return seconds and dateutc


def utc2tai(date):
    """ converts datetime.datetime UTC to datetime.datetime TAI.
        Works only on dates later than0o1.01.1972.

    Example
        >>> utc2tai(datetime(1992, 6, 2, 8, 6, 43))
        datetime.datetime(1992, 6, 2, 8, 7, 9)
    """
    seconds = __tai_seconds(to_utc_tz(date))
    return seconds and (date + timedelta(0, seconds))


def utc2taisec(date):
    """ converts datetime.datetime UTC to TAI seconds since the 1958 epoch.
        Works only on dates later than0o1.01.1972.

    Example
        t = utc_datetime(2015, 9, 1, 17, 45, 0)
        t1=tai.utc2taisec(t)
        t1
        Out[6]: 1819820736.0
        tai.taisec2utc(t1)
        Out[7]: datetime.datetime(2015, 9, 1, 17, 45, tzinfo=<UTC>)
    """
    # note this implies date must already be UTC-tz-aware
    seconds = __tai_seconds(date.replace(tzinfo=pytz.UTC))
    return seconds and (date - unix_epoch).total_seconds() + seconds + tai_unix_offset


def taisec2utc(tai):
    """ returns a datetime.datetime (UTC) object from a tai time in seconds.
        Works only on dates after0o1.01.1972.
        If the tai time includes fractional seconds, the fractional part is
        carried straight across into the python datetime microseconds field.

        Args
            tai: TAI time in integer seconds since 1958-01-01.

        Returns
            a datetime.datetime object at UTC

        Example
        >>> taisec2utc(1400000000.0)
        datetime.datetime(2002, 5, 13, 16, 52, 48)
    """
    st = time.gmtime(tai - tai_unix_offset)
    micro = min(999999, int(round(1000000 * (tai % 1))))
    dt = utc_datetime(st.tm_year, st.tm_mon, st.tm_mday,
                                     st.tm_hour, st.tm_min, st.tm_sec, micro)
    return tai2utc(dt)


if __name__ == '__main__':
    import doctest
    doctest.testmod()


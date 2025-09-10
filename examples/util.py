import xarray as xr
import warnings
import matplotlib as mpl
from matplotlib import pyplot as plt, dates as mdates, ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

def format_axes(ax, xaxis='on', yaxis='on', time=True):
    '''
    Format the abcissa and ordinate axes

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axes to be formatted
    time : bool
        If true, format the x-axis with dates
    xaxis, yaxis : str
        Indicate how the axes should be formatted. Options are:
        ('on', 'time', 'off'). If 'time', the ConciseDateFormatter is applied
        If 'off', the axis label and ticklabels are suppressed. If 'on', the
        default settings are used
    '''
    
    # All x-axes should be formatted with time
    if time:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    if xaxis == 'off':
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    if ax.get_yscale() == 'log':
        locmaj = ticker.LogLocator(base=10.0)
        ax.yaxis.set_major_locator(locmaj)
        
        locmin = ticker.LogLocator(base=10.0, subs=(0.3, 0.6, 0.9)) 
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    if yaxis == 'off':
        ax.set_yticklabels([])
        ax.set_ylabel('')


def add_legend(ax, labels=None, handles=None,
               corner='NE', outside=False, horizontal=False,
               labelspacing=0.5):
    '''
    Add a legend to the axes. Legend elements will have the same color as the
    lines that they label.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to which the legend is attached.
    lines : list of `matplotlib.lines.Line2D`
        The line elements that the legend format should match
    corner : str
        The bounding box of the legend will be tied to this corner:
        ('NE', 'NW', 'SE', 'SW')
    outside : bool
        The bounding box will extend outside the plot
    horizontal : bool
        The legend items will be placed in columns (side-by-side) instead of
        rows (stacked vertically)
    labelspacing : float
        Vertical space between legend labels, in font-size units
    '''
    
    if handles is None:
        handles = ax.get_lines()
    if labels is None:
        labels = [handle.get_label() for handle in handles]
    
    if horizontal:
        ncol = len(handles)
        columnspacing = 0.5
    else:
        ncol = 1
        columnspacing = 0.0
    
    if corner == 'N':
        bbox_to_anchor = (0.5, 1)
        loc = 'lower center' if outside else 'upper center'
    elif corner == 'NE':
        bbox_to_anchor = (1, 1)
        loc = 'upper left' if outside else 'upper right'
    elif corner == 'E':
        bbox_to_anchor = (0, 0.5)
        loc = 'center left' if outside else 'center right'
    elif corner == 'SE':
        bbox_to_anchor = (1, 0)
        loc = 'lower left' if outside else 'lower right'
    elif corner == 'S':
        bbox_to_anchor = (0.5, 0)
        loc = 'upper center' if outside else 'lower center'
    elif corner == 'SW':
        bbox_to_anchor = (0, 0)
        loc = 'lower right' if outside else 'lower left'
    elif corner == 'W':
        bbox_to_anchor = (0, 0.5)
        loc = 'center right' if outside else 'center left'
    elif corner == 'NW':
        bbox_to_anchor = (0, 1)
        loc = 'upper right' if outside else 'upper left'
    else:
        raise ValueError('corner must be a cardinal direction '
                         '(NW, N, NE, E, SE, S, SW, W), not {0}'
                         .format(corner))

    # Ignore the warning
    # The label '_di' of <matplotlib.lines.Line2D object at 0x141a00890>
    # starts with '_'. It is thus excluded from the legend.
    #
    # Alternately
    # warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        leg = ax.legend(handles, labels,
                        bbox_to_anchor=bbox_to_anchor,
                        borderaxespad=0.0,
                        columnspacing=columnspacing,
                        handlelength=1,
                        handletextpad=0.25,
                        labelspacing=labelspacing,
                        loc=loc,
                        ncol=ncol)
    
    # Turn the frame off but still add it to the legend
    leg.get_frame().set_alpha(0)
    
    # Make the lines and labels have the same color
    #   - Labels that start with "_" are hidden from the legend
    #   - Empty labels are not added to the legend
    texts = leg.get_texts()
    idx = 0
    for handle in handles:
        label = handle.get_label()
        if (label != '') and (label[0] != '_'):
            # Scatter plots are PathCollections
            #   - https://stackoverflow.com/questions/47908429/get-color-of-a-scatter-point
            if isinstance(handle, mpl.collections.PathCollection):
                color = handle.to_rgba(0)
            else:
                color = handle.get_color()
            
            texts[idx].set_color(color)
            idx += 1


def add_colorbar(ax, im):
    '''
    Add a colorbar to the axes.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to which the colorbar is attached.
    im : `matplotlib.axes.Axes.pcolorfast`
        The image that the colorbar will represent.
    '''
    cbaxes = inset_axes(ax,
                        width='2%', height='100%', loc=4,
                        bbox_to_anchor=(0, 0, 1.05, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cb.ax.minorticks_on()
    
    return cb

def _get_labels(da, label):
    
    # Add legend label
    if label is None:
        try:
            # Set the label for each line so that they can
            # be returned by Legend.get_legend_handles_labels()
            label = da.attrs['label']
        except KeyError:
            pass

    if not isinstance(label, list):
        label = [label]
    
    return label


def plot(data, ax=None, labels=None, legend=True, 
         title='', xaxis='on', xlabel=None, ylabel=None, **kwargs):
    
    # Make sure we can iterate over the data arrays,
    # not the data within them
    if isinstance(data, xr.DataArray):
        data = [data]
    if labels is None:
        labels = [None]*len(data)
    
    # Get a set of axes in which to plot
    if ax is None:
        ax = plt.axes()
    
    # Plot each data array
    lines = []
    for da, label in zip(data, labels):
        da_lines = da.plot(ax=ax)
    
        da_labels = _get_labels(da, label)
        for da_line, da_label in zip(da_lines, da_labels):
            da_line.set_label(da_label)
    
        lines.append(*da_lines)
    
    # Annotate axes
    ax.set_title(title)
    if xaxis == 'on':
        ax.set_xlabel(xlabel)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    if ylabel is None:
        try:
            ax.set_ylabel('{0}\n({1})'.format(data[0].attrs['ylabel'],
                                              data[0].attrs['units'])
                          )
        except KeyError:
            pass
    else:
        ax.set_ylabel(ylabel)
    
    # Add a legend
    if legend:
        # Create the legend outside the right-most axes
        leg = ax.legend(bbox_to_anchor=(1.05, 1),
                        borderaxespad=0.0,
                        frameon=False,
                        handlelength=0,
                        handletextpad=0,
                        loc='upper left')
        
        # Color the text the same as the lines
        for line, text in zip(lines, leg.get_texts()):
            text.set_color(line.get_color())
    
    return ax
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

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
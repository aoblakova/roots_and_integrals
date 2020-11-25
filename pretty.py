import matplotlib.pyplot as plt

utgreen = (0.302, 0.717, 0.341)
utred = (0.925, 0.11, 0.141)
utyellow = (1., 0.827, 0.341)


def top(wid_in_cm, hei_times):
    """Create a figure with width 'wid_in_cm' and height to width ration of 'hei_times'.
    Return axes, line width, and font size of 9 points."""
    inch = 2.54
    wid = wid_in_cm / inch
    hei = wid * hei_times
    points_in_inch = 72  # there are 72 points in a cm
    pt_size = 0.35146  # mm
    fontsize_9pt = 9 * pt_size * points_in_inch / (inch * 10)
    plt.rc('text', usetex=True)
    fig = plt.figure(1, figsize=(wid, hei))
    fig.subplots_adjust(0, 0, 1, 1)
    lw = 1
    ax = plt.axes()
    return ax, lw, fontsize_9pt


def pretty_float(x):
    """Format a float to no trailing zeroes."""
    return ('%f' % x).rstrip('0').rstrip('.')


def bottom_axes(ax, fontsize, x1, x2, y1, y2, xlabel, ylabel, file_name):
    """Put axes' labels, limits and save the figure."""
    plt.xlabel(xlabel, fontsize=fontsize * 0.9)
    plt.ylabel(ylabel, fontsize=fontsize * 0.9)

    plt.ylim([y1, y2])
    plt.xlim([x1, x2])
    ax.tick_params(axis='both', which='major', labelsize=0.8 * fontsize, direction="in")
    plt.setp(ax.spines.values(), linewidth=1)

    plt.savefig(file_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

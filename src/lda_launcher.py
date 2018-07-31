from __future__ import division

import sys
sys.path.insert(0, '../external_modules')
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
from parser import *
import serie2GAF

def getGASF(series, gaf_type = 'GASF', rescale = 'Zero'):
    '''Get all GASF from a <list> of segments.'''
    l = []
    for s in series:
        img, _ = serie2GAF.serie2GAF(list(s), GAF_type = gaf_type, rescale_type = rescale) #[(-1,1),(0,1)])
        l.append(img)
    return l


def gridGAFPlotting(data, suptitle="GAF Segments", alpha = 0.1, plt_win_func=False, fig_size=(12,12), labels=""):
    """Plots the data in a grid of plots.
    Args:
        data (list): the list of data to be used.
        title (str): grid title.
        columnToPlot: the column in the data to be plotted.
    """
    
    grid_side_size = int(round(np.sqrt(len(data))))
    fig, axes = None, None
    
    if isinstance(fig_size,tuple ):
        fig, axes = plt.subplots(grid_side_size, grid_side_size, figsize=fig_size)
    else:
        fig, axes = plt.subplots(grid_side_size, grid_side_size)
    
    fig.patch.set_alpha(alpha)

    count = 0
    for i, row in enumerate(axes):
        for j in range(grid_side_size):
            if count >= len(data):
                fig.delaxes(row[j])
            else:
                if not len(data[count]):
                    print "WARN: Windows #{} is empty!".format(count)
                    
                row[j].set_xticks(())
                row[j].set_yticks(())
                row[j].set_title("{}".format(count), fontsize=8, fontweight="bold")
                row[j].imshow(data[count])
                count += 1
                
    #fig.suptitle(suptitle, fontsize=21)
    fig.subplots_adjust(hspace=0.7, wspace=0.2)
    plt.draw()

def plotComposedGraph(new_segs=[]):
    n = 13 #-> number of windows
    offset = 0
    fig = plt.figure(figsize=(22, 7))
    fig.patch.set_alpha(0.1)
    for i in range(n):
        # display original
        ax = plt.subplot(4, n, i+1)
        ax.grid()
        ax.plot(serie2GAF.rescale(new_segs[i+offset]))
        ax.set_title("Original", fontweight="bold")
        

        # plot original segment signal
        ax = plt.subplot(4, n, i+1 + n, projection='polar')
        s = serie2GAF.rescale(new_segs[i+offset])
        ax.plot(np.arccos(s), range(len(s)), lw=2)
        ttl = ax.title
        ttl.set_position([.5, 1.25])
        ax.set_title("Polar", fontweight="bold")
        ax.set_rgrids(range(10,len(s),20), angle=250, fontsize=5)

        # display reconstruction
        ax = plt.subplot(4, n, i+1 + 2*n)
        g = [list(new_segs[i+offset])]
        g = getGASF(g)
        ax.grid()
        ax.imshow(g[0])
        ax.set_title("GASF", fontweight="bold")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(4, n, i+1 + 3*n)
        retrieved = serie2GAF.getSeriesFromGAF(g[0])
        ax.grid()
        ax.plot(retrieved)
        ax.set_title("Restored", fontweight="bold")
        
    plt.tight_layout()
    #plt.savefig('gasfpipeline.eps')


if __name__ == '__main__':
	
    windows = []

    new_segs = []

    X = new_parse("../inputs/log_file.txt")
    for d in X:
        for c in d.values():
            if len(c) >= 9:
                windows.append(c)
                new_segs.append(c)


    for i in range(len(new_segs)):
        new_segs[i] = np.array(new_segs[i])

    plotComposedGraph(new_segs)
    #gridGAFPlotting(getGASF(windows))
    plt.show()

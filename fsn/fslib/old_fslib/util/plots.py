__author__ = 'griver'
from matplotlib import pyplot as plt
import matplotlib.colors as cls

colors = list(cls.cnames)

class PlotBuilder(object):
    figure = None
    sp = []

    def create_figure(self, gridx = 1, gridy = 1):
        self.figure = plt.figure()
        self.sp = []
        id = 1
        for i in xrange(0, gridx * gridy):
            ax = self.figure.add_subplot(gridx, gridy, i + 1)
            self.sp.append(ax)

    # funcs are triples (y_value_array, 'line_style', label_name)
    def plot_curves(self, plt_id, x_axis, func, *funcs):
        cid = 0
        maxx = len(colors)
        lines = []
        tmp = self.sp[plt_id].plot(x_axis, func[0], func[1], color=colors[cid % maxx], label=func[2])    # func[1], label=func[2])
        lines.append(tmp[0])
        for f in funcs:
            cid += 1
            tmp = self.sp[plt_id].plot(x_axis, f[0], f[1], color=colors[cid % maxx], label=f[2])  # f[1], label=f[2])
            lines.append(tmp[0])

        # Now add the legend with some customizations.
        #if True: return

        legend = self.sp[plt_id].legend(loc='lower left', shadow=True)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1)  # the legend line width

        return

    # funcs triples  of type: (yvalues, label_name)
    # where xranges is sequence of (xmin, xwidth)
    # xlim = (xmin, xmax)
    # ylim = (ymin, ymax)
    def plot_bars(self, plt_id, xlim, ylim, thresh,  *funcs):
        if len(funcs) == 0:
            return

        ax = self.sp[plt_id]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ywidth = ylim[1] - ylim[0]
        ymin =  ylim[0]
        bar_width = float(ywidth) / len(funcs)
        id_list = range(0, len(funcs))

        for i in id_list:
            #print(brocken_bars[i][0])
            #print((bar_width*i, bar_width))
            #print("-----------------------------")
            bb = self._yvalue_to_bar(funcs[i][0], thresh)
            ax.broken_barh(bb, (bar_width*i, bar_width), facecolors='black')

        ticks_list = map(lambda i: bar_width* i + bar_width/2.0 + ymin, id_list)
        #print(ticks_list)
        ax.set_yticks(ticks_list)
        ax.set_yticklabels(zip(*funcs)[-1])
        ax.grid(True)

    def _yvalue_to_bar(self, yvalues, thresh):
        start = None
        l = None
        broken_bar = []
        for i in xrange(0, len(yvalues)):
            if start is not None:
                l += 1
                if yvalues[i] < thresh:
                    broken_bar.append((start, l))
                    start = None

            elif yvalues[i] >= thresh:
                start = i
                l = 0

        if start is not None:
            broken_bar.append((start, l))
        return broken_bar


    def get_subplot_ax(self, plt_id):
        return self.sp[plt_id]

    def show(self):
        plt.show()
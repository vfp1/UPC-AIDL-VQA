#!/usr/bin/env python

import pandas as pd
import matplotlib
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF

class VQA_reporting(object):
    """
    A class to report VQA results
    """

    def report(self, uuid_run, epoch, bsize, subset, loss, VGG_w, met, opt, lr, ts):


        df = pd.DataFrame()
        df['Hyperparams'] = ["Epoch", "Batch Size", "Subset", "Loss", "VGG weights",
                             "Metrics", "Optimizer", "Learning rate", "Test size"]
        df[uuid_run] = [epoch, bsize, subset, loss, VGG_w, met, opt, lr, ts]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.set_font('arial', 'B', 12)
        pdf.cell(60)
        pdf.cell(75, 10, "UPC AIDL report for VQA", 0, 2, 'C')
        pdf.cell(90, 10, " ", 0, 2, 'C')
        pdf.cell(-40)
        pdf.cell(50, 10, 'Question', 1, 0, 'C')
        pdf.cell(40, 10, 'Charles', 1, 0, 'C')
        pdf.cell(40, 10, 'Mike', 1, 2, 'C')
        pdf.cell(-90)
        pdf.set_font('arial', '', 12)

        for i in range(0, len(df)):
            pdf.cell(50, 10, '%s' % (df['Hyperparams'].ix[i]), 1, 0, 'C')
            pdf.cell(40, 10, '%s' % (str(df.uuid_run.ix[i])), 1, 0, 'C')
            pdf.cell(-90)
        pdf.cell(90, 10, " ", 0, 2, 'C')
        pdf.cell(-30)
        pdf.image('./barchart.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
        pdf.image('./barchart.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
        pdf.image('./barchart.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
        pdf.image('./barchart.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
        pdf.output('./test.pdf', 'F')
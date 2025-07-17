#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import string
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

letters = string.ascii_lowercase

def mpl_style():
    plt.rcParams['font.size']           = 18
    plt.rcParams['font.weight']         = 'bold'
    plt.rcParams['axes.titleweight']    = 'bold'
    plt.rcParams['axes.labelweight']    = 'bold'
    plt.rcParams['axes.xmargin']        = 0
    plt.rcParams['axes.titlesize']      = 'x-large'
mpl_style()

def my_xticks(sDate,eDate,ax,radar_ax=False,labels=True,short_labels=False,
                fmt='%d %b',fontdict=None,plot_axvline=True):
    if fontdict is None:
        fontdict = {'weight': 'bold', 'size':mpl.rcParams['ytick.labelsize']}
    xticks      = []
    xticklabels = []
    curr_date   = sDate
    while curr_date < eDate:
        if radar_ax:
            xpos    = get_x_coords(curr_date,sDate,eDate)
        else:
            xpos    = curr_date
        xticks.append(xpos)
        xticklabels.append('')
        curr_date += datetime.timedelta(days=1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Define xtick label positions here.
    # Days of month to produce a xtick label.
    doms    = [1,15]

    curr_date   = sDate
    ytransaxes = mpl.transforms.blended_transform_factory(ax.transData,ax.transAxes)
    while curr_date < eDate:
        if curr_date.day in doms:
            if radar_ax:
                xpos    = get_x_coords(curr_date,sDate,eDate)
            else:
                xpos    = curr_date

            if plot_axvline:
                axvline = ax.axvline(xpos,-0.015,color='k')
                axvline.set_clip_on(False)

            if labels:
                ypos    = -0.025
                txt     = curr_date.strftime(fmt)
                ax.text(xpos,ypos,txt,transform=ytransaxes,
                        ha='left', va='top',rotation=0,
                        fontdict=fontdict)
            if short_labels:    
                if curr_date.day == 1:
                    ypos    = -0.030
                    txt     = curr_date.strftime('%b %Y')
                    ax.text(xpos,ypos,txt,transform=ytransaxes,
                            ha='left', va='top',rotation=0,
                            fontdict=fontdict)
                    ax.axvline(xpos,lw=2,zorder=5000,color='0.6',ls='--')
        curr_date += datetime.timedelta(days=1)

    xmax    = (eDate - sDate).total_seconds() / (86400.)
    if radar_ax:
        ax.set_xlim(0,xmax)
    else:
        ax.set_xlim(sDate,sDate+datetime.timedelta(days=xmax))

def fmt_xaxis(ax,xlim=None,label=True):
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))
    ax.set_xlabel('Time [UTC]')
    ax.set_xlim(xlim)

def adjust_axes(ax_0,ax_1):
    """
    Force geospace environment axes to line up with histogram
    axes even though it doesn't have a color bar.
    """
    ax_0_pos    = list(ax_0.get_position().bounds)
    ax_1_pos    = list(ax_1.get_position().bounds)
    ax_0_pos[2] = ax_1_pos[2]
    ax_0.set_position(ax_0_pos)

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

def curve_combo_plot(result_dct, cb_pad=0.125,
                     output_dir=os.path.join('output', 'daily_plots')):
    """
    Revised combo plot with 11 panels:
        1. (a) Raw rarr (duplicated)
        2. (b) Raw rarr (duplicated)
        3. (c) Blurred arr with quantile medians
        4. (d) arr + detected edge + variation + region lines
        5. (e) arr + 1st sinfit used for detrending
        6. (f) Detrended edge (detected edge - poly fit)
        7. (g) Bandpass-filtered edge (use data_detrend here)
        8. (h) 2nd sinfit over bandpassed edge (reuse sin_fit here)
        9. (i) Text box with 2nd sinfit params (reuse p0_sin_fit)
       10. (j) arr + 2nd sinfit added back (poly_fit + sin_fit)
    """
    md = result_dct.get('metaData')
    date = md.get('date')
    xlim = md.get('xlim')
    winlim = md.get('winlim')
    fitWinLim = md.get('fitWinLim')
    qs = md.get('qs')

    rarr  = result_dct.get('rawArr')
    barr  = result_dct.get('intArr')
    arr = result_dct.get('spotArr')
    med_lines = result_dct.get('med_lines')
    edge_0 = result_dct.get('000_detectedEdge')
    sin_fit = result_dct.get('sin_fit')
    poly_fit = result_dct.get('poly_fit')
    stability = result_dct.get('stability')
    data_detrend = result_dct.get('data_detrend')
    p0_sin_fit = result_dct.get('p0_sin_fit')

    print(rarr)
    print(arr)
    print(barr)
    ranges_km = arr.coords['ranges_km']
    arr_times = [pd.Timestamp(x) for x in arr.coords['datetimes'].values]

    if data_detrend is None:
        data_detrend = edge_0 - poly_fit
        result_dct['data_detrend'] = data_detrend

    letters = list('abcdefghijk')
    fig = plt.figure(figsize=(19, 55))
    gs = gridspec.GridSpec(11, 2, width_ratios=[20, 0.5], height_ratios=[1]*11)
    axs = []
    axcbs = [None] * 11

    qs_vals = [float(col) for col in med_lines.columns if col != 'Time']
    med_lines_vals = [med_lines[q].values for q in qs_vals]

    # Panel (a) - Raw rarr heatmap (first duplicate)
    ax = fig.add_subplot(gs[0, 0])
    axs.append(ax)
    ax_cb = fig.add_subplot(gs[0, 1])
    axcbs[0] = ax_cb
    mpbl = ax.pcolormesh(arr_times, ranges_km, rarr.T, cmap='plasma', shading='nearest', rasterized=True, antialiased=False)
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Raw Data')
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(250, 2000)

    # Panel (c) — blurred arr with quantile medians
    ax = fig.add_subplot(gs[1, 0])
    axs.append(ax)
    ax_cb = fig.add_subplot(gs[1, 1])
    axcbs[1] = ax_cb
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma', shading='nearest', rasterized=True, antialiased=False)
#    for q, line in zip(qs_vals, med_lines_vals):
#        ax.plot(arr_times, line, label=f'Quantile {q}')
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Gaussian Filtered Data')
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(250, 2000)
    
    # Panel (b) - Raw rarr heatmap (second duplicate)
    ax = fig.add_subplot(gs[2, 0])
    axs.append(ax)
    ax_cb = fig.add_subplot(gs[2, 1])
    axcbs[2] = ax_cb
    mpbl = ax.pcolormesh(arr_times, ranges_km, barr.T, cmap='plasma', shading='nearest', rasterized=True, antialiased=False)
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='8bit Rescale Data')
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(250, 2000)

    # Panel (d) — arr + detected edge + stability + regions
    ax = fig.add_subplot(gs[3, 0])
    axs.append(ax)
    ax_cb = fig.add_subplot(gs[3, 1])
    axcbs[3] = ax_cb
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma', shading='nearest', rasterized=True, antialiased=False)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Blurred Data')
    ax2 = ax.twinx()
    ax2.plot(stability.index, stability, lw=2, color='0.5')
    ax2.grid(False)
    ax2.set_ylabel('Edge Coef. of Variation')
    for wl in winlim:
        ax.axvline(wl, color='0.8', ls='--', lw=2)
    for wl in fitWinLim:
        ax.axvline(wl, color='lime', ls='--', lw=2)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(250, 2000)

    # Panel (e) — arr + 1st sin fit
    ax = fig.add_subplot(gs[4, 0])
    axs.append(ax)
    ax_cb = fig.add_subplot(gs[4, 1])
    axcbs[4] = ax_cb
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma', shading='nearest', rasterized=True, antialiased=False)
    ax.plot(poly_fit.index, poly_fit, label='1st Sin Fit', color='white', lw=3, ls='--')
    ax.plot(arr_times, edge_0, lw=2, label='Detected Edge', color='cyan')
    for wl in fitWinLim:
        ax.axvline(wl, color='lime', ls='--', lw=2)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Blurred Data')
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(250, 2000)

    # Panel (f) — Detrended edge
    ax = fig.add_subplot(gs[5, 0])
    axs.append(ax)
    filtered = edge_0 - poly_fit
    ax.plot(filtered.index, filtered, label='Detrended Edge')
    for wl in fitWinLim:
        ax.axvline(wl, color='lime', ls='--', lw=2)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='x-small', ncols=4)

    # Panel (g) — Bandpass-filtered edge
    ax = fig.add_subplot(gs[6, 0])
    axs.append(ax)
    ax.plot(data_detrend.index, data_detrend, label='Bandpass Filtered')
    for wl in fitWinLim:
        ax.axvline(wl, color='lime', ls='--', lw=2)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='x-small', ncols=4)

    # Panel (h) — 2nd Sinfit
    ax = fig.add_subplot(gs[7, 0])
    axs.append(ax)
    ax.plot(data_detrend.index, data_detrend, label='Bandpass Filtered (simulated)')
    ax.plot(sin_fit.index, sin_fit, label='2nd Sin Fit (reuse 1st)', color='red', lw=3, ls='--')
    for wl in fitWinLim:
        ax.axvline(wl, color='lime', ls='--', lw=2)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='x-small', ncols=4)

    # Panel (i) — textbox with sin fit params
    ax = fig.add_subplot(gs[8, 0])
    axs.append(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fontdict = {'weight': 'normal', 'family': 'monospace'}
    txt = ['2nd Sinusoid Fit (simulated)']
    if p0_sin_fit:
        for key, val in p0_sin_fit.items():
            if key == 'r2':
                txt.append(f'{key}: {val:.2f}')
            elif key != 'selected':
                txt.append(f'{key}: {val:.1f}')
    ax.text(0.01, 0.95, '\n'.join(txt), fontdict=fontdict, va='top')

    # Panel (j) — arr + 2nd sin fit added back
    ax = fig.add_subplot(gs[9, 0])
    axs.append(ax)
    ax_cb = fig.add_subplot(gs[9, 1])
    axcbs[9] = ax_cb
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma', shading='nearest', rasterized=True, antialiased=False)
    ax.plot(sin_fit.index, poly_fit + sin_fit, label='Final Sin Fit', color='white', lw=3, ls='--')
    for wl in fitWinLim:
        ax.axvline(wl, color='lime', ls='--', lw=2)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Blurred Data')
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(250, 2000)
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)

    # Label all panels
    for ax_inx, ax in enumerate(axs):
        lbl = f'({letters[ax_inx]})'
        ax.set_title(lbl, loc='left')

    # Metadata title on top-most panel
    meta_title = []
    if md.get('freq_str') is not None:
        meta_title.append(md.get('freq_str'))
    region = md.get('region')
    if region:
        if region == 'NA':
            region = 'North America'
        meta_title.append(region)
    datasets = md.get('datasets')
    if datasets:
        meta_title.append(str(datasets))
    axs[0].set_title('\n'.join(meta_title), loc='right', fontdict={'size': 'x-small'})

    # Align all panels with panel (a)
    for ax_inx, ax in enumerate(axs):
        if ax_inx == 0:
            continue
        adjust_axes(ax, axs[0])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    date_str = date.strftime('%Y%m%d')
    png_fname = f'{date_str}_curveCombo.png'
    png_fpath = os.path.join(output_dir, png_fname)
    print(f'   Saving: {png_fpath}')
    fig.tight_layout()
    fig.savefig(png_fpath, bbox_inches='tight')
    plt.close()

    return result_dct



def sin_fit_key_params_to_csv(all_results,output_dir='output'):
    """
    Generate a CSV with sin fit parameters for an entire season.
    """

    sDate   = min(all_results.keys())
    eDate   = max(all_results.keys())

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = eDate.strftime('%Y%m%d')

    # Create parameter dataframe.
    params = []
    params.append('T_hr')
    params.append('amplitude_km')
    params.append('agree')
    
    df_lst = []
    df_inx = []
    for date,results in all_results.items():
        if results is None:
            continue

        p0_sin_fit = results.get('p0_sin_fit')
        tmp = {}
        for param in params:
            tmp[param] = p0_sin_fit.get(param,np.nan)

        df_lst.append(tmp)
        df_inx.append(date)

    df          = pd.DataFrame(df_lst,index=df_inx)
    # Force amplitudes to be positive.
    df.loc[:,'amplitude_km']    = np.abs(df['amplitude_km'])

    # Set non-LSTID parameters to NaN
    csv_fname   = '{!s}-{!s}_sinFit.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df.to_csv(csv_fpath)

def plot_sin_fit_analysis(all_results,
                          T_hr_vmin=0,T_hr_vmax=5,T_hr_cmap='rainbow',
                          output_dir='output'):
    """
    Plot an analysis of the sin fits for the entire season.
    """

    sDate   = min(all_results.keys())
    eDate   = max(all_results.keys())

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = eDate.strftime('%Y%m%d')
    png_fname   = '{!s}-{!s}_sinFitAnalysis.png'.format(sDate_str,eDate_str)
    png_fpath   = os.path.join(output_dir,png_fname)

    # Create parameter dataframe.
    params = []
    params.append('T_hr')
    params.append('amplitude_km')
    params.append('phase_hr')
    params.append('offset_km')
    params.append('slope_kmph')
    params.append('r2')
    params.append('T_hr_guess')
    params.append('selected')

    df_lst = []
    df_inx = []
    for date,results in all_results.items():
        if results is None:
            continue

        all_sin_fits = results.get('all_sin_fits')
        for p0_sin_fit in all_sin_fits:
            tmp = {}
            for param in params:
                if param in ['selected']:
                    tmp[param] = p0_sin_fit.get(param,False)
                else:
                    tmp[param] = p0_sin_fit.get(param,np.nan)
            
            # Get the start and end times of the good fit period.
            fitWinLim   =  results['metaData']['fitWinLim']
            tmp['fitStart'] = fitWinLim[0]
            tmp['fitEnd']   = fitWinLim[1]

            df_lst.append(tmp)
            df_inx.append(date)

    df                = pd.DataFrame(df_lst,index = df_inx)
    # Calculate the duration in hours of the good fit period.
    df['duration_hr'] = (df['fitEnd'] - df['fitStart']).apply(lambda x: x.total_seconds()/3600.)
    df_sel            = df[df.selected].copy() # Data frame with fits that have been selected as good.

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = eDate.strftime('%Y%m%d')
    csv_fname   = '{!s}-{!s}_allSinFits.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df.to_csv(csv_fpath)

    csv_fname   = '{!s}-{!s}_selectedSinFits.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df_sel.to_csv(csv_fpath)

    # Plotting #############################
    nrows   = 4
    ncols   = 1
    ax_inx  = 0
    axs     = []

    cbar_info = {} # Keep track of colorbar info in a dictionary to plot at the end after fig.tight_layout() because of issues with cbar placement.

    figsize = (30,nrows*6.5)
    fig     = plt.figure(figsize=figsize)

    # ax with LSTID Amplitude Analysis #############################################
    prmds   = {}
    prmds['amplitude_km'] = prmd = {}
    prmd['title']   = 'Ham Radio TID Amplitude'
    prmd['label']   = 'Amplitude [km]'
    prmd['vmin']    = 10
    prmd['vmax']    = 60

    prmds['T_hr'] = prmd = {}
    prmd['title']   = 'Ham Radio TID Period'
    prmd['label']   = 'Period [hr]'
    prmd['vmin']    = 0
    prmd['vmax']    = 5

    prmds['r2'] = prmd = {}
    prmd['title']   = 'Ham Radio Fit $r^2$'
    prmd['label']   = '$r^2$'
    prmd['vmin']    = 0
    prmd['vmax']    = 1

    for param in ['amplitude_km','T_hr','r2']:
        prmd            = prmds.get(param)
        title           = prmd.get('title',param)
        label           = prmd.get('label',param)

        ax_inx  += 1
        ax              = fig.add_subplot(nrows,ncols,ax_inx)
        axs.append(ax)

        xx              = df_sel.index
        yy_raw          = df_sel[param]
        rolling_days    = 5
        title           = '{!s} ({!s} Day Rolling Mean)'.format(title,rolling_days)
        yy              = df_sel[param].rolling(rolling_days,center=True).mean()

        vmin            = prmd.get('vmin',np.nanmin(yy))
        vmax            = prmd.get('vmax',np.nanmax(yy))

        cmap            = mpl.colormaps.get_cmap(T_hr_cmap)
        norm            = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        mpbl            = mpl.cm.ScalarMappable(norm,cmap)
        color           = mpbl.to_rgba(yy)
        ax.plot(xx,yy_raw,color='0.5',label='Raw Data')
        ax.plot(xx,yy,color='blue',lw=3,label='{!s} Day Rolling Mean'.format(rolling_days))
        ax.scatter(xx,yy,marker='o',c=color)
        ax.legend(loc='upper right',ncols=2)

        trans           = mpl.transforms.blended_transform_factory( ax.transData, ax.transAxes)
        ax.bar(xx,1,width=1,color=color,align='edge',zorder=-1,transform=trans,alpha=0.5)

        cbar_info[ax_inx] = cbd = {}
        cbd['ax']       = ax
        cbd['label']    = label
        cbd['mpbl']     = mpbl

        ylabel_fontdict         = {'weight': 'bold', 'size':24}
        ax.set_ylabel(label,fontdict=ylabel_fontdict)
        my_xticks(sDate,eDate,ax,fmt='%d %b')
        ltr = '({!s}) '.format(letters[ax_inx-1])
        ax.set_title(ltr+title, loc='left')

    # ax with LSTID T_hr Fitting Analysis ##########################################    
    ax_inx  += 1
    ax      = fig.add_subplot(nrows,ncols,ax_inx)
    axs.append(ax)
    ax_0    = ax


    xx      = df.index
    yy      = df.T_hr
    color   = df.T_hr_guess
    r2      = df.r2.values
    r2[r2 < 0]  = 0
    alpha   = r2
    mpbl    = ax.scatter(xx,yy,c=color,alpha=alpha,marker='o',
                         vmin=T_hr_vmin,vmax=T_hr_vmax,cmap=T_hr_cmap)

    ax.scatter(df_sel.index,df_sel.T_hr,c=df_sel.T_hr_guess,ec='black',
                         marker='o',label='Selected Fit',
                         vmin=T_hr_vmin,vmax=T_hr_vmax,cmap=T_hr_cmap)
    cbar_info[ax_inx] = cbd = {}
    cbd['ax']       = ax
    cbd['label']    = 'T_hr Guess'
    cbd['mpbl']     = mpbl

    ax.legend(loc='upper right')
    ax.set_ylim(0,10)
    ax.set_ylabel('T_hr Fit')
    my_xticks(sDate,eDate,ax,labels=(ax_inx==nrows))

    fig.tight_layout()

#    # Account for colorbars and line up all axes.
    for ax_inx, cbd in cbar_info.items():
        ax_pos      = cbd['ax'].get_position()
                    # [left, bottom,       width, height]
        cbar_pos    = [1.025,  ax_pos.p0[1], 0.02,   ax_pos.height] 
        cax         = fig.add_axes(cbar_pos)
        fig.colorbar(cbd['mpbl'],label=cbd['label'],cax=cax)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('   Saving: {!s}'.format(png_fpath))
    fig.savefig(png_fpath,bbox_inches='tight')

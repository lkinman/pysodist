# -*- coding: utf-8 -*-
"""
@author: Laurel Kinman <lkinman@mit.edu> jhdavislab.org
@version: 0.0.4
"""
import warnings

warnings.filterwarnings("ignore")
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from os import path
import argparse
import pickle

warnings.filterwarnings("default")
import pysodist
import pysodist.utils.skyline_report_defs as defs
from pysodist.utils import utilities

log = utilities.log
import sys
import pysodist.commands.plot_spectra as ps
from scipy import optimize
from multiprocessing import Pool
from itertools import chain, combinations
import pysodist.utils.skyline_report_defs as defs

def add_args(parser):
    parser.add_argument('workdir', type = str, default = './', help = 'Working directory')
    parser.add_argument('--spectrassr', type = int, required = True, help = 'Spectra SSR to filter scans by')
    parser.add_argument('--ssr', type = int, required = True, help = 'Gaussian fitting SSR to filter peptides by')
    parser.add_argument('--threads', type = int, required = True, help = 'Number of threads to use in parallel')
    parser.add_argument('--filt', type = str, default = 'AMP_F', help = 'Value to fit and filter by (AMP_U, AMP_L, AMP_F, or sum). Default is AMP_F')
    parser.add_argument('--plotfilt', type = bool, default = True, help = 'Whether to plot retained and filtered-out spectra')
    parser.add_argument('--subset', action = 'store_true', help = 'Run filtration on a random subset of data to test parameters')
    return parser

def gaussian(x, a, c, s, w):
    return a*np.exp(-(x + s)**2/(2*w**2)) + c 

def fit_amps(peptide_df, amp_col):
    x = pd.to_numeric(peptide_df['retention_time']).values

    if type(amp_col) == str:
        y = peptide_df[amp_col].values
    if type(amp_col) == list:
        y = peptide_df[amp_col[0]]
        for i in amp_col:
            y = y + peptide_df[i] 
        y = y.values
    
    c_guess = y.min() 
    if len(y) > 1:
        if max(y) <= 2*np.sort(y)[-2]:
            s_guess = -1*x[np.argmax(y)]
            a_guess = y.max() - y.min()
        elif max(y) > 2*np.sort(y)[-2]:
            s_guess = -1*x[np.where(y == np.sort(y)[-2])[0][0]]
            a_guess = np.sort(y)[-2] - y.min()
    else:
        s_guess = -1*(x.min() + (x.max() - x.min())/2)
        a_guess = y.max() - y.min()
    w_guess = (x.max() - x.min())/3
    guesses = [a_guess, c_guess, s_guess, w_guess]
    bounds = ([0, -np.inf, -1*(x.max()), 0], [np.inf, np.inf, -1*(x.min()), x.max() - x.min()])


    popt, pcov = optimize.curve_fit(gaussian, x, y, p0 = guesses, bounds = bounds)
    return x, y, popt, pcov

def calc_ssr(x, y, popt):
    resid_y = gaussian(x, *popt)
    y_mean = np.mean(y)
    ssr_adj = sum(((resid_y - y)/y_mean)**2)/len(resid_y)
    return ssr_adj

def run_fit_amps(inputs):
    fails = []
    pep_df, fit_col, fig_output, ft, wd = inputs
    pep = pep_df['pep'].unique()[0]
    popt = None
    adj_ssr = None
    
    try:
        nrows = int(np.ceil((len(pep_df) + 1)/2))

        page_height = 4*nrows
        fig, ax = plt.subplots(nrows, 2, figsize = (12, page_height), tight_layout = True)
        ax = ax.flatten()

        x_vals, y_vals, popt, __ = fit_amps(pep_df, fit_col)
        adj_ssr = calc_ssr(x_vals, y_vals, popt)
        fitx = np.linspace(x_vals.min(), x_vals.max(), 100)
        fity = gaussian(fitx, *popt)

        ax[0].plot(fitx, fity, '--r')
        ax[0].scatter(x_vals, y_vals, s = 10)

        ax[0].set_title(pep)

        for j in range(0, len(pep_df)):
            id_row = pep_df.iloc[j]
            spectral_dict = ps.read_spectra(id_row, ft, working_path = wd)
            ps.plot_fit(spectral_dict, ax[j + 1], info_box = False)

        fig.savefig(fig_output + pep + '.png')
        plt.close()
    except TypeError:
        fails.append(pep)
    except RuntimeError:
        fails.append(pep)
    except ValueError:
        fails.append(pep)
        
    return pep, popt, adj_ssr, fails

def plot_fits_spectra(df, fit_col, fit_output, fit_output_cols, fig_output, ft, num_threads, wd):
    all_peps = df['pep'].unique()
    pep_dfs_list = []
    
    for pep in all_peps:
        sub = df[df['pep'] == pep]
        pep_dfs_list.append([sub, fit_col, fig_output, ft, wd])
    
    pool = Pool(num_threads)
    fits = pool.map(run_fit_amps, pep_dfs_list)
    
    for i in fits:
        pep, popt, adj_ssr = i[0:3]
        inds = df[df['pep'] == pep].index
        fit_output.loc[inds, fit_output_cols[1:]] = popt
        fit_output.loc[inds, fit_output_cols[0]] = adj_ssr
        
    fail_list = [i[3] for i in fits if len(i[3]) > 0]
    fail_list = list(chain(*fail_list))
    
    return fit_output, fail_list

def calc_spectra_ssr(inputs):
    df, row, ft, wd = inputs
    spectra = ps.read_spectra(df.loc[row], ft, working_path = wd)
    spectra_ssr = np.sum((spectra['data']['resid']/np.mean(spectra['data']['intensity']))**2)/len(spectra['data']['resid'])
    return row, spectra_ssr

def plot_filt_spectra(inputs):
    pep_df, output, ft, wd = inputs
    pep = pep_df['pep'].unique()[0]
    nrows = int(np.ceil(len(pep_df)/2))
    fig, ax = plt.subplots(nrows, 2, figsize = (12, 4*nrows))
    ax = ax.flatten()

    i = 0
    for spectra in pep_df.index:
        spectral_dict = ps.read_spectra(pep_df.loc[spectra], ft, working_path = wd)
        ps.plot_fit(spectral_dict, ax[i], info_box = False)
        i = i + 1

    fig.savefig(output + pep + '.png')
    plt.close()
    return

def run_filtration(spectrassr, ssr, filt, wd, threads, subset = False, plotfilt = False):
    configs_dict = {'spectrassr': spectrassr, 'ssr': ssr, 'filt': filt}
    ft = wd + [i for i in os.listdir(wd) if '_isodist_fits' in i][0] + '/'
    out = utilities.check_dir(wd + 'gaussian_fits/', make = True)
    fitname = [i for i in os.listdir(wd) if '_isodist_outputs' in i][0]
    isodist_result = wd + fitname + '/' + fitname.split('_isodist')[0] + '_output.csv'

    id_result = ps.parse_isodist_csv(isodist_result)
    ampcols = id_result.columns[id_result.columns.str.contains('AMP')]
    ratio_cols = []
    for i in combinations(ampcols, 2):
        col1, col2 = [j.split('AMP_')[-1] for j in i]
        id_result.loc[:, f'{col1}/{col2}'] = id_result[i[0]]/id_result[i[1]].round(3)
        ratio_cols.append(f'{col2}/{col2}')

    nonsum_id = id_result[id_result['retention_time'] != 'SUM']
    print('Filtering by GW...')
    gw_filt = nonsum_id[(nonsum_id['gw'] >= 0.01) & (nonsum_id['gw'] <= 1)] ###FIGURE OUT WHETHER WE NEED TO KEEP THIS AND/OR MAKE IT MORE STANDARD/RIGOROUS

    if subset:
        rand_inds = np.random.choice(gw_filt.index, int(len(gw_filt)/10))
        gw_filt = gw_filt.loc[rand_inds, :]

    print('Calculating spectra SSR...')
    spectra_ssr_list = []
    for i in gw_filt.index:
        spectra_ssr_list.append([gw_filt, i, ft, wd])

    pool = Pool(threads)
    spectra_ssr_results = pool.map(calc_spectra_ssr, spectra_ssr_list)

    for i in spectra_ssr_results:
        row, spectra_ssr = i
        gw_filt.loc[row, 'Spectra SSR'] = spectra_ssr
    if not subset:
        gw_filt.to_csv(out + 'gw_filt.csv')
    
    gw_filt = gw_filt[gw_filt['Spectra SSR'] <= spectrassr] 
    spectra = []
    all_peps = gw_filt['pep'].unique()
    for pep in all_peps:
        pep_df = gw_filt[gw_filt['pep'] == pep]
        pep_meds = pep_df.median()
        
        drop_inds = []
        for i in pep_df.index:
            for j in ampcols:
                if (pep_df.loc[i, j] < 0.1*pep_meds[j]) or (pep_df.loc[i, j] > 20*pep_meds[j]):
                    drop_inds.append(i)

        sub_pep = pep_df.drop(drop_inds)
        if len(sub_pep) > 3:
            for i in range(0, len(sub_pep.index)):
                spectra.append(sub_pep.index.to_list()[i])

    gw_amp_filt = gw_filt.loc[spectra]
    if not subset:
        gw_amp_filt.to_csv(out + 'gw_amp_filt.csv')

    fit_results = pd.DataFrame(index = gw_amp_filt.index)

    abbrev = str(filt).split('_')[1].lower()
    result_cols = [f'SSR_gaussian_{abbrev}', f'Gaussian_{abbrev}_a', f'Gaussian_{abbrev}_c', f'Gaussian_{abbrev}_s', f'Gaussian_{abbrev}_w']
    storefits = utilities.check_dir(f'{out}gaussian_{abbrev}', make = True)
    print('fitting and plotting ' + str(filt) + '...')
    fit_results, failed_fits = plot_fits_spectra(gw_amp_filt, filt, fit_results, result_cols, storefits, ft, threads, wd)
    
    print('writing fit_results file...')
    fit_results.to_csv(out + 'fit_results.csv')

    print('writing failed fits...')
    fail_file = out + 'failed_fits.txt'
    with open(fail_file, 'w') as f:
        f.write('Failed fits\n')
        for j in failed_fits:
            f.write('\t' + str(j) + '\n')

    fit_results = fit_results.fillna(np.inf)
    abbrev_filt = fit_results[fit_results[f'SSR_gaussian_{abbrev}'] <= ssr].index 

    print('filtering by specified measure...')
    s_col = f'Gaussian_{abbrev}_s'
    w_col = f'Gaussian_{abbrev}_w'

    filt_peps = []
    for pep in gw_amp_filt.loc[abbrev_filt, 'pep'].unique():
        pep_df = gw_amp_filt[gw_amp_filt['pep'] == pep]
        pep_s = fit_results.loc[pep_df.index[0], s_col]
        rts = pd.to_numeric(pep_df['retention_time'])
        rt_lb = rts.min() 
        rt_ub = rts.max() 
        
        if -1*pep_s <= rt_ub and -1*pep_s >= rt_lb:
            filt_peps.append(pep)
            
    sub_pepfilt = gw_amp_filt[gw_amp_filt['pep'].isin(filt_peps)]

    filt_scans = []
    for spectra in sub_pepfilt.index:
        rt = np.float(sub_pepfilt.loc[spectra, 'retention_time'])
        s = fit_results.loc[spectra, s_col]
        w = fit_results.loc[spectra, w_col]
        rt_lb = -1*s - 0.75*w
        rt_ub = -1*s + 0.75*w
        if rt >= rt_lb and rt <= rt_ub:
            filt_scans.append(spectra)
        
    sub_pepscanfilt = gw_amp_filt.loc[filt_scans]
    filtered_out = nonsum_id.loc[~nonsum_id.index.isin(filt_scans)]

    sub_pepscanfilt.to_csv(out + 'sub_pepscanfilt.csv')
    filtered_out.to_csv(out + 'filtered_out.csv')
    with open(out + 'configs_dict.pkl', 'wb') as f:
        pickle.dump(configs_dict, f)

    if plotfilt:
        utilities.check_dir(out + 'pepscanfilt/', make = True)
        utilities.check_dir(out + 'filtered_out/', make = True)

        print('plotting filtered and filtered-out spectra...')
        pepscanfilt_dfs = []
        for pep in sub_pepscanfilt['pep'].unique():
            sub = sub_pepscanfilt[sub_pepscanfilt['pep'] == pep]
            pepscanfilt_dfs.append([sub, out + 'pepscanfilt/', ft, wd]) 
        pool = Pool(threads)
        pool.map(plot_filt_spectra, pepscanfilt_dfs)

        filteredout_dfs = []
        for pep  in filtered_out['pep'].unique():
            sub = filtered_out[filtered_out['pep'] == pep]
            filteredout_dfs.append([sub, out + 'filtered_out/', ft, wd])
        pool = Pool(threads)
        pool.map(plot_filt_spectra, filteredout_dfs)
    return

def main(args):
    run_filtration(args.spectrassr, args.ssr, args.filt, args.workdir, args.threads, args.subset, args.plotfilt)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())
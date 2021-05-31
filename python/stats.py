# Viet Hoang
# ID: 260789801

# Statistics
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import seaborn as sns

# Misc.
from dipy.data import pjoin
import pickle
import warnings
warnings.filterwarnings('ignore')

# I/O
def io(base):
    
    with open(pjoin(base, 'pickle/ad.pkl'), 'rb') as handle:
        lh_ad_df, rh_ad_df = pickle.load(handle)
    
    with open(pjoin(base, 'pickle/fa.pkl'), 'rb') as handle:
        lh_fa_df, rh_fa_df = pickle.load(handle)
    
    with open(pjoin(base, 'pickle/md.pkl'), 'rb') as handle:
        lh_md_df, rh_md_df = pickle.load(handle)
    
    with open(pjoin(base, 'pickle/rd.pkl'), 'rb') as handle:
        lh_rd_df, rh_rd_df = pickle.load(handle)

    with open(pjoin(base, 'pickle/lesion.pkl'), 'rb') as handle:
        lh_lesion_df, rh_lesion_df = pickle.load(handle)

    lh = [lh_ad_df, lh_fa_df, lh_md_df, lh_rd_df, lh_lesion_df]
    rh = [rh_ad_df, rh_fa_df, rh_md_df, rh_rd_df, rh_lesion_df]

    for i in range(len(lh)):
        lh[i].fillna(method='bfill', inplace=True)
        rh[i].fillna(method='bfill', inplace=True)

    return (lh, rh)


def io_global(base):

    with open(pjoin(base, 'pickle/ad_global.pkl'), 'rb') as handle:
        ad = pickle.load(handle)
    
    with open(pjoin(base, 'pickle/fa_global.pkl'), 'rb') as handle:
        fa = pickle.load(handle)
    
    with open(pjoin(base, 'pickle/md_global.pkl'), 'rb') as handle:
        md = pickle.load(handle)
    
    with open(pjoin(base, 'pickle/rd_global.pkl'), 'rb') as handle:
        rd = pickle.load(handle)

    sp = [ad[0], fa[0], md[0], rd[0]]
    rr = [ad[1], fa[1], md[1], rd[1]]
    hc = [ad[2], fa[2], md[2], rd[2]]
    
    return [sp, rr, hc]


def plot_global(metric):

    names = ['AD','FA', 'MD', 'RD']
    fig, axes = plt.subplots(2,2,figsize=(16,9))

    for i in range(len(names)):
        plt.subplot(2,2,i+1)

        groups = [metric[0][i], metric[1][i], metric[2][i]]
        sns.boxplot(data=groups, width=0.25)
        
        plt.xticks([0, 1, 2], ['SP', 'RR', 'HC'])
        plt.title(names[i])

        _, pvals = ttest_ind(metric[0][i], metric[2][i])
        print(pvals)

    plt.show()


# Plots the along tract profile for a single metric (example for Methods)
# We will extract data relating to tuv_020
def plot_raw_data_single(base, df_lst, tract, patient):

    names = ['AD','FA', 'MD', 'RD']
    fig, axes = plt.subplots(2,2,figsize=(16,9))

    for i in range(len(names)):
        axes = plt.subplot(2,2,i+1)
        metric = df_lst[i][df_lst[i].filter(like=patient).columns]

        plt.plot(metric, color='royalblue', marker='.')

        plt.xlabel('Percentage along tract')
        axes.xaxis.set_major_formatter(PercentFormatter())
        
        plt.ylabel(names[i])
        
    plt.suptitle(tract, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    plt.show()


def plot_lesions(base, df_lst, tract):
    fig = plt.figure(1, (8,5))
    ax = fig.add_subplot(1,1,1)


    df = df_lst[4]        
    for column in df:        
        color = 'bisque'
        if '_HC' in column:
            color = 'lavender'
        
        plt.plot(df[column], color=color)
    
    
    hc_avg = df[df.filter(like='HC').columns].mean(axis=1)
    plt.plot(hc_avg, color='royalblue', marker='.', label='Healthy control')

    ms_avg = df[df.filter(regex='RR|SP').columns].mean(axis=1)
    plt.plot(ms_avg, color='orange', marker='.', label='Multiple Sclerosis')

    plt.xlabel('Percentage along tract')
    ax.xaxis.set_major_formatter(PercentFormatter())
        
    plt.ylabel('Proportion of streamlines intersecting a lesion')
    plt.legend()

    plt.title(tract, fontsize=14, fontweight='bold')
    
    plt.show()     


# plots the along tract profile for all metrics
def plot_raw_data_all(base, df_lst, tract):
    
    names = ['AD','FA', 'MD', 'RD']
    fig, axes = plt.subplots(2,2,figsize=(16,9))

    for i in range(len(names)):
        axes = plt.subplot(2,2,i+1)

        df = df_lst[i]        
        for column in df:        
            color = 'bisque'
            if '_HC' in column:
                color = 'lavender'
        
            plt.plot(df[column], color=color)
    
    
        hc_avg = df[df.filter(like='HC').columns].mean(axis=1)
        plt.plot(hc_avg, color='royalblue', marker='.', label='Healthy control')

        ms_avg = df[df.filter(regex='RR|SP').columns].mean(axis=1)
        plt.plot(ms_avg, color='orange', marker='.', label='Multiple Sclerosis')

        plt.xlabel('Percentage along tract')
        axes.xaxis.set_major_formatter(PercentFormatter())
        
        plt.ylabel(names[i])
        plt.legend()
    
    plt.suptitle(tract, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    plt.show()


# Performs independent along-tract t-tests and plots the results
def ttest_all(base, df_lst, tract, corrected=True):

    names = ['AD','FA', 'MD', 'RD']
    fig, axes = plt.subplots(2,2,figsize=(16,9))

    for i in range(len(names)):
        axes = plt.subplot(2,2,i+1)
        
        df = df_lst[i]
        
        hc = df[df.filter(like='HC').columns].to_numpy(copy=True)
        ms = df[df.filter(regex='^((?!HC).)*$').columns].to_numpy(copy=True)
        _, pvals = ttest_ind(hc, ms, axis=1)        
        _, corrected_pvals,_,_ = multipletests(pvals, alpha=0.05, 
                                               method='fdr_bh', is_sorted=False, 
                                               returnsorted=False)

        if corrected:
            plt.plot(corrected_pvals, color='royalblue', marker='s', markersize=3)
        else:
            plt.plot(pvals, color='royalblue', marker='s', markersize=3)
        
        plt.axhline(y=0.05, color='r', linewidth=0.5, label='p=0.05')
        plt.xlabel('Percentage along tract')
        axes.xaxis.set_major_formatter(PercentFormatter())
        plt.ylabel('p-value')
        plt.yscale('log')
        plt.legend()
        plt.title(names[i], fontweight='bold')


    plt.suptitle(tract, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    plt.show() 
           

def normalize(df_lst):
    
    norm = []
    for i in range(len(df_lst)-1):
        df = df_lst[i]
        hc_avg = df[df.filter(like='HC').columns].mean(axis=1)

        df = df.subtract(hc_avg,axis=0)
        norm.append(df)
    
    norm.append(df_lst[4])

    return norm


def plot_norm(base, df_lst, tract):
    names = ['AD','FA', 'MD', 'RD']
    fig, axes = plt.subplots(2,2,figsize=(16,9))

    for i in range(len(names)):
        axes = plt.subplot(2,2,i+1)
        
        df = df_lst[i]
        for column in df:        
            color = 'lavender'
            if '_RR' in column:
                color = 'bisque'
            elif '_SP' in column:
                color = 'lightpink'
        
            plt.plot(df[column], color=color)


        hc_avg = df[df.filter(like='HC').columns].mean(axis=1)
        plt.plot(hc_avg, color='royalblue', marker='.', label='Healthy control')

        spms_avg = df[df.filter(like='SP').columns].mean(axis=1)
        plt.plot(spms_avg, color='orange', marker='.', label='SPMS')

        rrms_avg = df[df.filter(like='RR').columns].mean(axis=1)
        plt.plot(rrms_avg, color='crimson', marker='.', label='RRMS')             

        plt.xlabel('Percentage along tract')
        axes.xaxis.set_major_formatter(PercentFormatter())
        
        plt.ylabel(names[i])
        plt.legend()


    plt.suptitle(tract, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    plt.show() 
    

def plot_lesion_norm_allms(base, df_lst, lesion):

    lesion = lesion[lesion.filter(regex='SP|RR').columns].to_numpy(copy=True)

    names = ['AD','FA', 'MD', 'RD']
    fig, axes = plt.subplots(2,2,figsize=(16,9))

    for i in range(len(names)):
        axes = plt.subplot(2,2,i+1)
        
        df = df_lst[i]    
        ms = df[df.filter(regex='SP|RR').columns].to_numpy(copy=True)    

        avg_lesion = []
        avg_nolesion = [] 


        for row in range(len(ms)):
        
            with_lesion = []
            no_lesion = []
        
            for col in range(len(ms[row,:])):
            
                # Deal with SPs first
                if lesion[row, col] > 0.05:
                    with_lesion.append(ms[row,col])
                else:
                    no_lesion.append(ms[row,col])
            
            
            avg_lesion.append(np.mean(with_lesion))
            avg_nolesion.append(np.mean(no_lesion))

        a = np.array(avg_nolesion)
        b = np.array(avg_lesion)
        x = a[~np.isnan(a)]
        y= b[~np.isnan(b)]
        
        _, pvals = ttest_ind(x, y)
        print(pvals)


        groups = [avg_nolesion, avg_lesion]
        sns.boxplot(data=groups, width=0.25)
        
        plt.xticks([0, 1], ['Regions of CST with no lesions', 'Regions of the CST with lesions'])
        plt.title(names[i])
    
    plt.show()


def plot_lesion_norm(base, df, lesion):
    
    spms = df[df.filter(like='SP').columns].to_numpy(copy=True)
    rrms = df[df.filter(like='RR').columns].to_numpy(copy=True)
    lesion_sp = lesion[lesion.filter(like='SP').columns].to_numpy(copy=True)
    lesion_rr = lesion[lesion.filter(like='RR').columns].to_numpy(copy=True)

    avg_lesion_sp = []
    avg_lesion_rr = []
    avg_nolesion_sp = []
    avg_nolesion_rr = [] 

    
    # Deal with spms first
    for row in range(len(spms)):
        
        with_lesion = []
        no_lesion = []
        
        for col in range(len(spms[row,:])):
            
            # Deal with SPs first
            if lesion_sp[row, col] > 0.1:
                with_lesion.append(spms[row,col])
            else:
                no_lesion.append(spms[row,col])
            
            
        
        avg_lesion_sp.append(np.mean(with_lesion))
        avg_nolesion_sp.append(np.mean(no_lesion))
    
    for row in range(len(rrms)):
        
        with_lesion = []
        no_lesion = []
        
        for col in range(len(rrms[row,:])):
            
            # Deal with SPs first
            if lesion_rr[row, col] > 0.1:
                with_lesion.append(rrms[row,col])
            else:
                no_lesion.append(rrms[row,col])
            
            
        print(len(with_lesion))
        avg_lesion_rr.append(np.mean(with_lesion))
        avg_nolesion_rr.append(np.mean(no_lesion))
    
    
    plt.plot(avg_lesion_sp, label='sp lesion', marker='.', linestyle='None')
    plt.plot(avg_nolesion_sp, label='sp no lesion', marker='.', linestyle='None')
    plt.plot(avg_lesion_rr, label='rr lesion', marker='.', linestyle='None')
    plt.plot(avg_nolesion_rr, label='rr no lesion', marker='.', linestyle='None')
    plt.legend()
    plt.show()
    

def main():
    
    base = "/home/viethoang/RudkoLab/Honours"    
    lh, rh = io(base)
    
    for df in lh:
        df.drop('aad_028_m00_SP', axis=1, inplace=True)
    
    for df in rh:
        df.drop('aad_028_m00_SP', axis=1, inplace=True)

    global_metric = io_global(base)
    plot_global(global_metric)

    plot_raw_data_single(base, lh, 'Left Corticospinal Tract', 'aah')

    plot_lesions(base, lh, 'Left Corticospinal Tract')
    plot_lesions(base, rh, 'Right Corticospinal Tract')
    
    plot_raw_data_all(base, lh, 'Left Corticospinal Tract')
    plot_raw_data_all(base, rh, 'Right Corticospinal Tract')

    ttest_all(base, lh, 'Left Corticospinal Tract', corrected=True)
    ttest_all(base, rh, 'Right Corticospinal Tract', corrected=True)

    ttest_all(base, lh, corrected=False)
    ttest_all(base, rh, corrected=False)

    # Normalize all the values except for the lesion. Normalization occurs by 
    # subtracting all the MS patients metric values by the average healthy
    # control metric value. Lesions are not normalized
    lh_norm = normalize(lh)
    rh_norm = normalize(rh)

    plot_norm(base, lh_norm, 'Left Corticospinal Tract')
    plot_norm(base, rh_norm, 'Right Corticospinal Tract')

    ttest_all(base, lh, 'Left Corticospinal Tract', corrected=True)
    ttest_all(base, rh, 'Right Corticospinal Tract', corrected=True)
    
    plot_lesion_norm_allms(base, lh_norm, lh_norm[-1])


if __name__ == "__main__":
    main()
import numpy as np    
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D           
import myfunctions as mf   
from tqdm import tqdm                               
import pandas as pd
import os

def plot_data(exp, OS, offset=0, save=True):

    progress_bar = tqdm(total=5, desc=f'{exp} drawing plots', position=offset, leave=False)
    length, heigth = 5, 3.5
    fig = plt.figure(figsize=(3*length, 5*heigth), dpi=150)
    subfigs = fig.subfigures(5, 1, hspace=0.3)
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette(sns.color_palette(['#3cb44b', '#bfef45']))
    palette2 = ['#e6194B', '#f58231', '#ffe119'] # ['#bce4b5', '#56b567', '#05712f'] #
    palette3 = ['#000075', '#4363d8', '#42d4f4'] # ['#fdc692', '#f67824', '#ad3803'] # 

    df = pd.read_csv(os.path.join(mf.OS_path(exp, OS), 'motion_properties.csv'))
    time_axis = np.arange(len(np.unique(df['t'])))/20

    # creating 'total' dataframe
    grouped_df = df.groupby('t').size().reset_index(name='N')
    df_tot = pd.merge(df[['t']], grouped_df, on='t', how='left').drop_duplicates()
    df_tot['V'] = df.groupby('t')['V'].sum().values
    df_tot['V/N'] = df_tot['V'] / df_tot['N'] # np.maximum(df_tot['N'].values, 1)
    df_tot['dVdt'] = df.groupby('t')['dVdt'].sum().values
    df_tot.fillna(0, inplace=True)
    df_tot.reset_index(drop=True, inplace=True)

    # creating 'r' dataframe
    r_sect_list = ['Core', 'Intermediate', 'External']
    df_r = pd.DataFrame(columns=['t', 'r_sect', 'V/N'])
    df_r['t'] = np.repeat(df['t'].unique(), len(r_sect_list))
    df_r['r_sect'] = np.tile(r_sect_list, len(df['t'].unique()))
    grouped_N = df.groupby(['t', 'r_sect']).size().reset_index(name='N')
    grouped_V = df.groupby(['t', 'r_sect'])['V'].sum().reset_index(name='V')
    grouped_dVdt = df.groupby(['t', 'r_sect'])['dVdt'].sum().reset_index(name='dVdt')
    df_r = pd.merge(df_r, grouped_N, on=['t', 'r_sect'], how='left')
    df_r = pd.merge(df_r, grouped_V, on=['t', 'r_sect'], how='left')
    df_r = pd.merge(df_r, grouped_dVdt, on=['t', 'r_sect'], how='left')
    df_r.fillna(0, inplace=True)
    df_r['V/N'] = df_r['V'] / np.maximum(df_r['N'].values, 1)
    df_r.reset_index(drop=True, inplace=True)

    # creating 'z' dataframe
    z_sect_list = ['Bottom', 'Middle', 'Top']
    df_z = pd.DataFrame(columns=['t', 'z_sect', 'V/N'])
    df_z['t'] = np.repeat(df['t'].unique(), len(z_sect_list))
    df_z['z_sect'] = np.tile(z_sect_list, len(df['t'].unique()))
    grouped_N = df.groupby(['t', 'z_sect']).size().reset_index(name='N')
    grouped_V = df.groupby(['t', 'z_sect'])['V'].sum().reset_index(name='V')
    grouped_dVdt = df.groupby(['t', 'z_sect'])['dVdt'].sum().reset_index(name='dVdt')
    df_z = pd.merge(df_z, grouped_N, on=['t', 'z_sect'], how='left')
    df_z = pd.merge(df_z, grouped_V, on=['t', 'z_sect'], how='left')
    df_z = pd.merge(df_z, grouped_dVdt, on=['t', 'z_sect'], how='left')
    df_z.fillna(0, inplace=True)
    df_z['V/N'] = df_z['V'] / np.maximum(df_z['N'].values, 1)
    df_z.reset_index(drop=True, inplace=True)

    # Agglomerates total volume vs time
    subfigs[0].suptitle('Agglomerates total volume vs time', y=1.1, fontsize=14)
    axs = subfigs[0].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='V')
    sns.lineplot(ax=plt.twinx(ax=axs[0]), data=df_tot, x='t', y='N', color='#bfef45')
    axs[0].set_title('Whole battery')
    axs[0].legend(handles=[Line2D([], [], marker='_', color='#3cb44b', label='Volume [$mm^3$]'), Line2D([], [], marker='_', color='#bfef45', label='Number of agglomerates')], loc='upper left')
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='V', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='V', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Volume [$mm^3$]')
    progress_bar.update()

    # Agglomerates mean volume vs time
    subfigs[1].suptitle('Agglomerates average volume vs time', y=1.1, fontsize=14)
    axs = subfigs[1].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='V/N')
    axs[0].set_title('Whole battery')
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='V/N', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='V/N', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Volume [$mm^3$]')
    progress_bar.update()

    # Agglomerates total volume expansion rate vs time
    subfigs[2].suptitle('Agglomerates total volume expansion rate vs time', y=1.1, fontsize=14)
    axs = subfigs[2].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='dVdt')
    axs[0].set_title('Whole battery')
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='dVdt', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper right')
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='dVdt', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Volume expansion rate [$mm^3/s$]')
    progress_bar.update()

    # Agglomerates speed vs time
    subfigs[3].suptitle('Agglomerates speed vs time', y=1.1, fontsize=14)
    axs = subfigs[3].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df, x='t', y='v')
    axs[0].set_title('Modulus')
    sns.lineplot(ax=axs[1], data=df, x='t', y='vxy', color='#f58231')
    axs[1].set_title('$xy$ component')
    sns.lineplot(ax=axs[2], data=df, x='t', y='vz', color='#4363d8')
    axs[2].set_title('$z$ component')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Speed [$mm/s$]')
    progress_bar.update()

    # Agglomerates density vs time
    battery_volume = np.pi * (0.5 * mf.find_diameter(exp))**2 * 0.012

    subfigs[4].suptitle('Agglomerates density vs time', y=1.1, fontsize=14)
    axs = subfigs[4].subplots(1, 3, sharey=True)
    df_tot['N'] = df_tot['N'] / (battery_volume)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='N')
    axs[0].set_title('Whole battery')
    df_r.loc[df_r['r_sect'] == 'Core', 'N'] = df_r.loc[df_r['r_sect'] == 'Core', 'N'] / (battery_volume/9)
    df_r.loc[df_r['r_sect'] == 'Intermediate', 'N'] = df_r.loc[df_r['r_sect'] == 'Intermediate', 'N'] / (battery_volume*3/9)
    df_r.loc[df_r['r_sect'] == 'External', 'N'] = df_r.loc[df_r['r_sect'] == 'External', 'N'] / (battery_volume*5/9)
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='N', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper left')
    df_z['N'] = df_z['N'] / (battery_volume*3/9)
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='N', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper left')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        _ = ax.set_ylabel('Agglomerate density [cm$^{-3}$]')
    progress_bar.update()

    if save:
        fig.savefig(os.path.join(mf.OS_path(exp, OS), 'motion_properties.png'), dpi=300, bbox_inches='tight')

    progress_bar.close()
    return None
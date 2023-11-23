import numpy as np    
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D           
import myfunctions as mf   
from tqdm import tqdm                               
import pandas as pd
import os

def plot_data(exp, OS, offset, save):

    progress_bar = tqdm(total=5, desc=f'{exp} drawing plots', position=offset, leave=False)

    df = pd.read_csv(os.path.join(mf.OS_path(exp, OS), 'motion_properties.csv'))
    plt.style.use('seaborn-v0_8-paper')
    time_axis = np.arange(len(np.unique(df['t'])))/20
    heigth = 3.5
    length = 5
    fig = plt.figure(figsize=(3*length, 5*heigth), dpi=150)
    subfigs = fig.subfigures(5, 1, hspace=0.3)
    palette2 = ['#bce4b5', '#56b567', '#05712f']
    palette3 = ['#fdc692', '#f67824', '#ad3803']

    df_tot = pd.DataFrame(columns=['t', 'N', 'V', 'V/N', 'dVdt'])
    df_r = pd.DataFrame(columns=['t', 'N', 'V', 'V/N', 'dVdt', 'r_sect'])
    df_z = pd.DataFrame(columns=['t', 'N', 'V', 'V/N', 'dVdt', 'z_sect'])
    r_sect_list = ['Core', 'Intermediate', 'External']
    z_sect_list = ['Bottom', 'Middle', 'Top']
    for t in (np.unique(df['t'])):
        df_tot = pd.concat([df_tot, pd.DataFrame([[t, 0, 0, 0, 0]], columns=['t', 'N', 'V', 'dVdt'])], ignore_index=True)
        for r, z in zip(r_sect_list, z_sect_list):
            df_r = pd.concat([df_r, pd.DataFrame([[t, 0, 0, 0, 0, r]], columns=['t', 'N', 'V', 'dVdt', 'r_sect'])], ignore_index=True)
            df_z = pd.concat([df_z, pd.DataFrame([[t, 0, 0, 0, 0, z]], columns=['t', 'N', 'V', 'dVdt', 'z_sect'])], ignore_index=True)
    for i in range(len(df)):
        df_tot.loc[df_tot['t'] == df['t'][i], 'N'] += 1
        df_tot.loc[df_tot['t'] == df['t'][i], 'V'] += df['V'][i]
        df_tot.loc[df_tot['t'] == df['t'][i], 'dVdt'] += df['dVdt'][i]
        df_r.loc[(df_r['t'] == df['t'][i]) & (df_r['r_sect'] == df['r_sect'][i]), 'N'] += 1
        df_r.loc[(df_r['t'] == df['t'][i]) & (df_r['r_sect'] == df['r_sect'][i]), 'V'] += df['V'][i]
        df_r.loc[(df_r['t'] == df['t'][i]) & (df_r['r_sect'] == df['r_sect'][i]), 'dVdt'] += df['dVdt'][i]
        df_z.loc[(df_z['t'] == df['t'][i]) & (df_z['z_sect'] == df['z_sect'][i]), 'N'] += 1
        df_z.loc[(df_z['t'] == df['t'][i]) & (df_z['z_sect'] == df['z_sect'][i]), 'V'] += df['V'][i]
        df_z.loc[(df_z['t'] == df['t'][i]) & (df_z['z_sect'] == df['z_sect'][i]), 'dVdt'] += df['dVdt'][i]
    for t in (np.unique(df['t'])):
        df_tot.loc[df_tot['t'] == t, 'V/N'] = df_tot.loc[df_tot['t'] == t, 'N'] / df_tot.loc[df_tot['t'] == t, 'V']
        for r, z in zip(r_sect_list, z_sect_list):
            df_r.loc[(df_r['t'] == t) & (df_r['r_sect'] == r), 'V/N'] = df_r.loc[(df_r['t'] == t) & (df_r['r_sect'] == r), 'N'] / df_r.loc[(df_r['t'] == t) & (df_r['r_sect'] == r), 'V']
            df_z.loc[(df_z['t'] == t) & (df_z['z_sect'] == z), 'V/N'] = df_z.loc[(df_z['t'] == t) & (df_z['z_sect'] == z), 'N'] / df_z.loc[(df_z['t'] == t) & (df_z['z_sect'] == z), 'V']
    df_r.loc[df_r['r_sect'] == 'Intermediate', 'N'] = df_r.loc[df_r['r_sect'] == 'Intermediate', 'N'] / 3
    df_r.loc[df_r['r_sect'] == 'External', 'N'] = df_r.loc[df_r['r_sect'] == 'External', 'N'] / 5

    # VOLUME
    subfigs[0].suptitle('Agglomerates total volume vs time', y=1.1, fontsize=14)
    axs = subfigs[0].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='V', color='#1F77B4')
    sns.lineplot(ax=plt.twinx(ax=axs[0]), data=df_tot, x='t', y='N', color='#87C9FF', dashes=[(2, 2)])
    axs[0].set_title('Whole battery')
    axs[0].legend(handles=[Line2D([], [], marker='_', color='#1F77B4', label='Volume [$mm^3$]'), Line2D([], [], marker='_', color='#87C9FF', label='Number of agglomerates')], loc='upper left')
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

    # VOLUME/NUMBER
    subfigs[1].suptitle('Agglomerates total volume vs time', y=1.1, fontsize=14)
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

    # EXPANSION RATE
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

    # SPEED
    subfigs[3].suptitle('Agglomerates speed vs time', y=1.1, fontsize=14)
    axs = subfigs[3].subplots(1, 3, sharey=True)
    sns.lineplot(ax=axs[0], data=df, x='t', y='v')
    axs[0].set_title('Modulus')
    sns.lineplot(ax=axs[1], data=df, x='t', y='vxy', color='#bce4b5')
    axs[1].set_title('$xy$ component')
    sns.lineplot(ax=axs[2], data=df, x='t', y='vz', color='#fdc692')
    axs[2].set_title('$z$ component')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        ax.set_ylabel('Speed [$mm/s$]')
    progress_bar.update()

    # DENSITY
    subfigs[4].suptitle('Agglomerates density vs time', y=1.1, fontsize=14)
    axs = subfigs[4].subplots(1, 3)
    sns.lineplot(ax=axs[0], data=df_tot, x='t', y='N')
    axs[0].set_title('Whole battery')
    sns.lineplot(ax=axs[1], data=df_r, x='t', y='N', hue='r_sect', hue_order=r_sect_list, palette=palette2)
    axs[1].set_title('$r$ sections')
    axs[1].legend(loc='upper left')
    sns.lineplot(ax=axs[2], data=df_z, x='t', y='N', hue='z_sect', hue_order=z_sect_list, palette=palette3)
    axs[2].set_title('$z$ sections')
    axs[2].legend(loc='upper left')
    for ax in axs:
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel('Time [$s$]')
        _ = ax.set_ylabel('Agglomerate density [a.u.]')
    progress_bar.update()

    if save:
        fig.savefig(os.path.join(mf.OS_path(exp, OS), 'motion_properties.png'), dpi=300, bbox_inches='tight')

    progress_bar.close()
    return None
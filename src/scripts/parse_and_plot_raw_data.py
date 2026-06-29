#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a one-time-use script to parse the raw data files and calculate FRET for the
deadenylation kinetics assay.                                                                                        
Run script as: python parse_and_plot_raw_data.py parse_data.yaml                                
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import yaml
from icecream import ic


def main():
    config_params = yaml.safe_load(open(sys.argv[1],'r'))
    
    if config_params['Parse']['Parse data'] == True:
        df, output_file_name = parse_data(config_params)
    
    if config_params['Plot']['Plot data'] == True:
        if config_params['Parse']['Parse data'] == False:
            output_file_name = config_params['Parse']['Output file name']
            df = pd.read_csv(f"{output_file_name}.csv", header=0)
        plot_data(config_params, df, output_file_name)


def parse_data(config_params):

    output_file_name = config_params['Parse']['Output file name']
    data_to_load = config_params['Parse']['Data to load']
    time_arrays = np.transpose(np.array(pd.read_csv(config_params['Parse']['Time arrays'], header=0)))

    # Initialize an empty array to store FRET values
    fret_array = []
    norm_fret_array = []
    start_fret = []

    # Load CSV file
    global_norm = []
    global_min = []
    file_to_global_norm = []
    index_to_norm = []
    for i,file in enumerate(data_to_load):

        dataload = pd.read_csv(data_to_load[file]["Name"], header=None)

        # Remove columns with all NaN values
        nan_cols = []
        for i,x in enumerate(np.transpose(dataload.values)):
            if type(x[0]) != str:
                for j,y in enumerate(x):
                    if np.isnan(y):
                        app = 1
                    else:
                        app = 0
                        break
                if app == 1:
                    nan_cols.append(i)
        dataload = np.delete(dataload.values, nan_cols, axis=1)
        dataload = np.array(dataload)

        meas_num = dataload[6:,0]

        # Save row 2 of data starting at column 3 as enzyme_conc
        enzyme_conc = dataload[1][2:]
        unique_enzyme_conc = np.unique(enzyme_conc)

        # Save row 3 of data starting at column 3 as cap1_conc
        cap1_conc = dataload[2][2:]
        unique_cap1_conc = np.unique(cap1_conc)

        # Save row 4 of data starting at column 3 as rna_conc
        rna_conc = dataload[3][2:]
        unique_rna_conc = np.unique(rna_conc)

        # Save row 5 of data starting at column 3 as dna_conc
        dna_conc = dataload[4][2:]
        unique_dna_conc = np.unique(dna_conc)

        # Save first column of data as point from row 7 onwards
        points = np.transpose(dataload)[0][6:]

        # Convert points to integers
        points = points.astype(int)
        unique_points = np.unique(points)

        # Save row 6 of data starting at column 3 as time array index
        time_index = dataload[5][2:].astype(int)

        # Save column 2 of data as channels from row 7 onwards
        channels = np.transpose(dataload)[1][6:]
        unique_channels = np.unique(channels)
        all_IDD_idx = np.all([channels == 'DD'],axis=0)
        all_IDA_idx = np.all([channels == 'DA'],axis=0)
        all_IAA_idx = np.all([channels == 'AA'],axis=0)

        # Save data starting at row 7 and column 3 as data
        data = np.transpose(dataload[6:,2:])
        temp_fret_blank = []
        temp_fret_min = []
        temp_fret_array = []
        temp_enzyme_array = []
        tmp_rna = []
        tmp_dna = []
        tmp_enz = []
        tmp_time = []

        for dna in unique_dna_conc:
            if dna == 0:
                continue
            else:
                for rna in unique_rna_conc:
                    for cap1 in unique_cap1_conc:
                        dF = []
                        for enzyme in unique_enzyme_conc:
                            temp_dF = []
                            temp_data = data[np.all([enzyme_conc == enzyme, cap1_conc == cap1, rna_conc == rna, dna_conc == dna],axis=0 ),:]
                            temp_time_idx = time_index[np.all([enzyme_conc == enzyme, cap1_conc == cap1, rna_conc == rna, dna_conc == dna],axis=0 )]
                            temp_time = time_arrays[temp_time_idx-1]
                            if enzyme == 0:
                                blank_IDD = np.mean(temp_data[:,all_IDD_idx], axis=0)
                            for di, column in enumerate(temp_data):    
                                adj_IDA = []
                                IDD = column[all_IDD_idx]
                                IDA = column[all_IDA_idx]
                                IAA = column[all_IAA_idx]
                                
                                if config_params['Parse']['Subtract zero'] == True:
                                    adj_IDD = IDD - blank_IDD
                                    # adj_IDD = IDD/(blank_IDD/blank_IDD[0])
                                    adj_IDA = IDA/(IAA/IAA[0])
                                else:
                                    adj_IDD = IDD
                                    adj_IDA = IDA

                                FRET = adj_IDA/(adj_IDD+adj_IDA) # calculate FRET

                                enzyme = round(enzyme*1e6,2)/1e6 # convert to uM

                                if enzyme == 0:
                                    for j,fret in enumerate(FRET):
                                        temp_fret_blank.append(fret) # save all FRET values for 0 uM enzyme to calculate mean of first 3 time points
                                    fmax = np.mean(temp_fret_blank[0:3]) # calculate mean of first 3 time points for 0 uM enzyme
                                else:
                                    fmin_idx = np.argmin(FRET)
                                    fmin = FRET[fmin_idx]
                                    fminm1 = FRET[fmin_idx-1]
                                    if fmin_idx == len(FRET)-1:
                                        fminp1 = FRET[fmin_idx]
                                    else:
                                        fminp1 = FRET[fmin_idx+1]
                                    if np.abs(fmin - fminm1) > 0.05 or np.abs(fmin - fminp1) > 0.05: # if last lowest point differs by more than 0.05 from its neighbors, take the last point as low FRET
                                        # if np.abs(FRET[-1] - FRET[-2]) > 0.05: # if last two points differ by more than 0.05, take second last point as end FRET
                                        #     lowFRET = FRET[-2]
                                        # else:
                                        #     lowFRET = FRET[-1]
                                        lowFRET = FRET[-1]
                                    else:
                                        lowFRET = fmin
                                    
                                    dF.append(FRET[0]-lowFRET) # calculate dF for each enzyme concentration
                                # Append values to fret_array so that each row is [time, FRET, enzyme, rna, dna]
                                tmp_rna.append(rna)
                                tmp_dna.append(dna)
                                tmp_enz.append(float(f"{enzyme:.2e}"))
                                tmp_time.append(temp_time[di])    
                                temp_fret_array.append(list(FRET))
                                temp_enzyme_array.append(float(f"{enzyme:.3e}"))
                                for Fi,fret in enumerate(FRET):
                                    fret_array.append([temp_time[di][Fi],fret,enzyme,rna,dna])
                                    
        if config_params['Parse']['Normalize FRET'] == True:
            norm_to_dir = data_to_load[file]["Normalize to"].split(', ')
            if len(norm_to_dir) == 1:
                norm_to = norm_to_dir[0]
            else:
                norm_to_array = norm_to_dir
            mean_dF = np.mean(dF) # calculate mean dF across all enzyme concentrations (excluding 0 uM)
            std_dF = np.std(dF) # calculate std of dF across all enzyme concentrations (excluding 0 uM)
            for i, v in enumerate(dF): # remove outliers from dF by replacing any value less than mean - 2*std with the max dF
                if v < mean_dF - 2*std_dF:
                    dF[i] = np.max(dF)
            for ei in temp_enzyme_array:
                if ei == 0:
                    dF = np.concatenate(([np.max(dF)],dF)) # add mean dF to the beginning of the array for the 0 uM enzyme
            for ei,enzyme in enumerate(temp_enzyme_array): # normalize FRET for each enzyme concentration
                fmin = temp_fret_array[ei][0] - dF[ei] # fmin is the starting FRET - dF for that enzyme concentration
                norm_FRET = (temp_fret_array[ei] - fmin)/dF[ei] # normalize FRET to 0-1 range
                global_norm.append(dF[ei])
                global_min.append(fmin)
                if enzyme == 0:
                    norm_to = "self" # if 0 uM enzyme, must normalize to self
                else:
                    norm_to_index = np.where(unique_enzyme_conc == enzyme)[0][0]
                    norm_to = norm_to_array[norm_to_index-1] if len(norm_to_dir) > 1 else norm_to
                if norm_to == "self":
                    for Fi, fret in enumerate(norm_FRET):
                        index_to_norm.append(0)
                        norm_fret_array.append([tmp_time[ei][Fi],fret,tmp_enz[ei],tmp_rna[ei],tmp_dna[ei]])
                        if tmp_time[ei][Fi] == 0:
                            start_fret_val = fret
                        start_fret.append(start_fret_val)
                elif norm_to == "global":
                    if file not in file_to_global_norm:
                        file_to_global_norm.append(file)
                    for Fi, fret in enumerate(temp_fret_array[ei]):
                        norm_fret_array.append([tmp_time[ei][Fi],fret,tmp_enz[ei],tmp_rna[ei],tmp_dna[ei]])
                        index_to_norm.append(1)
                        # index_to_norm.append(len(norm_fret_array)-1)
                        if tmp_time[ei][Fi] == 0:
                            start_fret_val = fret
                        start_fret.append(start_fret_val)


    if config_params['Parse']['Normalize FRET'] == True and np.sum(index_to_norm) > 0:
            global_norm_mean = np.mean(global_norm)
            global_norm_std = np.std(global_norm)
            global_min_mean = np.mean(global_min)
            global_min_std = np.std(global_min)

            # Remove outliers from global_norm and global_min
            indices_to_keep = [i for i, v in enumerate(global_norm) if v >= global_norm_mean - global_norm_std]
            global_norm = [global_norm[i] for i in indices_to_keep]
            global_min = [global_min[i] for i in indices_to_keep]
            global_norm_mean = np.mean(global_norm)
            global_min_mean = np.mean(global_min)
            dF = global_norm_mean
            fmin = global_min_mean
            for i, val in enumerate(index_to_norm):
                if val == 1:
                    enzyme = norm_fret_array[i][2]
                    start_fret_val = (start_fret[i] - fmin) / dF # brings start value to 1 for global data
                    fret = norm_fret_array[i][1]
                    norm_fret_array[i][1] = (fret - fmin)/dF + (1 - start_fret_val)


    if config_params['Parse']['Normalize FRET'] == False:
        export_array = fret_array
    elif config_params['Parse']['Normalize FRET'] == True:
        export_array = norm_fret_array

    if config_params['Parse']['Subtract zero'] == True:
        # Remove rows where enzyme == 0
        export_array = [row for row in export_array if row[2] != 0]

    # Save fret_array as CSV file
    export_df = pd.DataFrame(export_array, columns=['Time', 'FRET','Enzyme','RNA', 'DNA'])
    df = export_df.sort_values(['RNA', 'Enzyme','Time'], ascending=[True, True, True], ignore_index=True)
    df.to_csv(f'{output_file_name}.csv', index=False)
    
    return df, output_file_name

def plot_data(config_params, df, output_file_name):

    ## Plot data ##
    protein = config_params['Sample name']
    rnas = df["RNA"].unique()
    enzymes = df['Enzyme'].unique()
    times = df["Time"].unique()

    mean_df = pd.DataFrame(columns=["Time", "mFRET", "Stdev", "RNA", "Enzyme"])

    # Determine the mean FRET and standard deviation for each RNA, enzyme, and time
    for rna in rnas:
        for enzyme in enzymes:
            for time in times:
                # Filter the dataframe based on the specified values
                filtered_df = df[(df["RNA"] == rna) & (df["Enzyme"] == enzyme) & (df["Time"] == time)]
                if filtered_df.empty:
                    continue
                else:
                    mean_fret = filtered_df["FRET"].mean()
                    stdev_fret = filtered_df["FRET"].std()
                    stdev_fret = stdev_fret if not np.isnan(stdev_fret) else 0
                    temp_df = pd.DataFrame({"Time": time, "mFRET": mean_fret, "Stdev": stdev_fret, "RNA": rna, "Enzyme": enzyme}, index=[0])
                    mean_df = pd.concat([mean_df if not mean_df.empty else None, 
                                        temp_df if not temp_df.empty else None], ignore_index=True)


    all_enzymes = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for enzyme in enzymes:
        if enzyme*1e+6 not in all_enzymes:
            all_enzymes.append(enzyme*1e+6)

    for ei in df['Enzyme']:
        if ei*1e+6 not in all_enzymes:
            all_enzymes.append(ei*1e+6)
    all_enzymes = np.sort(all_enzymes)

    points = len(all_enzymes)
    colormap = cm.jet(np.linspace(1, 0, points))        
    # ylim = [-0.2,1.2]
    yticks = np.linspace(0,1,6)

    len_rna = len(rnas)

    if config_params['Plot']['Plot mean data'] == True:
        plot_df = mean_df
        ylabel = "mean FRET"
    elif config_params['Plot']['Plot mean data'] == False:
        plot_df = df
        ylabel = "FRET"

    if len_rna == 1:
        data_fig, ax = plt.subplots(1,1,figsize=(2+len_rna*5,5)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
        max_time = round(max(plot_df['Time']),-3)
        if max_time <= 4000:
            max_time = 4000
        elif max_time <= 6000:
            max_time = 6000
        else:
            max_time = max_time
        if config_params['Plot']['Set x-axis limits'] == True:
            max_time = config_params['Plot']['Max time']
        
        xlim = [0-0.03*max_time, max_time+0.03*max_time]
        ylim = [-0.5, 1.5]
        xticks = np.linspace(0,max_time,6)
        for j, enzyme in enumerate(enzymes):

                filtered_df = plot_df[(plot_df["RNA"] == rna) & (plot_df["Enzyme"] == enzyme)]

                color_idx = np.where(all_enzymes == round(enzyme*1e6,1))[0]
                # color_idx = j
                line_label = f"{enzyme*1e6:.1f}"

                if config_params['Plot']['Plot mean data'] == True:
                    ax.errorbar(filtered_df["Time"],filtered_df["mFRET"],yerr=filtered_df["Stdev"],fmt='o',markersize=5,mfc=colormap[color_idx],
                                mec=colormap[color_idx],mew=1,capsize=0,capthick=1,ecolor=colormap[color_idx],label = line_label)
                elif config_params['Plot']['Plot mean data'] == False:
                    ax.scatter(filtered_df["Time"],filtered_df["FRET"],s=8,c=colormap[color_idx],label = line_label)

                ax.set_title(f"{protein} + {rna*1e9:.0f} nM RNA")
                ax.xaxis.set_ticks(xticks)
                # ax.yaxis.set_ticks(yticks)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(ylabel)
                legend_title = f"[Enzyme] (uM)"
                legend = ax.legend(title=legend_title,frameon=False, loc='upper left', fontsize=8, bbox_to_anchor=(1.05, 1))

    else:
        data_fig, axs = plt.subplots(len_rna,1,figsize=(7,len_rna*4)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
        for i, rna in enumerate(rnas):
            temp_data = plot_df[(plot_df["RNA"] == rna)]
            max_time = round(max(temp_data['Time']),-3)
            if max_time <= 4000:
                max_time = 4000
            elif max_time <= 8000:
                max_time = 8000
            xlim = [0-0.03*max_time, max_time+0.03*max_time]
            xticks = np.linspace(0,max_time,6)
            for j, enzyme in enumerate(enzymes):
                filtered_df = plot_df[(plot_df["RNA"] == rna) & (plot_df["Enzyme"] == enzyme)]
                color_idx = j
                line_label = f"{enzyme*1e6:.1f}"

                # FRET data plots
                if config_params['Plot']['Plot mean data'] == True:
                    axs[i].errorbar(filtered_df["Time"],filtered_df["mFRET"],yerr=filtered_df["Stdev"],fmt='o',markersize=5,
                                    mfc=colormap[color_idx],mec=colormap[color_idx],mew=1,capsize=0,capthick=1,
                                    ecolor=colormap[color_idx],label = line_label)
                elif config_params['Plot']['Plot mean data'] == False:
                    axs[i].scatter(filtered_df["Time"],filtered_df["FRET"],s=8,c=colormap[color_idx],label = line_label)

                axs[i].set_title(f"{protein} + {rna*1e9:.0f} nM RNA")
                axs[i].xaxis.set_ticks(xticks)
                # axs[i].yaxis.set_ticks(yticks)
                axs[i].set_xlim(xlim)
                # axs[i].set_ylim(ylim)
                axs[i].set_ylabel("mean FRET")
                if i == len_rna-1:
                    axs[i].set_xlabel("Time (s)")
                if i == 0:
                    legend_title = f"[Enzyme] (uM)"
                    legend = axs[i].legend(title=legend_title,frameon=False, loc='upper left', fontsize=8, bbox_to_anchor=(1.05, 1))

    data_fig.tight_layout()

    if config_params['Plot']['Save plot as']['PDF'] == True:
        data_fig.savefig(f"{output_file_name}_plotted.pdf", format='pdf')
    if config_params['Plot']['Save plot as']['PNG'] == True:   
        data_fig.savefig(f"{output_file_name}_plotted.png", format='png', transparent=True, dpi=1200)

    plt.close(data_fig)


main()
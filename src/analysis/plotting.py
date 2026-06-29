#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors: Robert W. Harkness, Rose M. Irwin
"""

import matplotlib as mpl
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from matplotlib import gridspec
import platform
import os


class PlotHandler:

    def __init__(self, experiments, kinetic_models, hybridization_models, resids, config_params):

        self.experiments = experiments
        self.kinetic_models = kinetic_models
        self.hybridization_models = hybridization_models
        self.residuals = resids
        self.sample_name = config_params['Sample name']
        max_time = round(np.max([np.max(v) for experiment in experiments for v in experiment.time]),-2)
        self.timesample = [int(round(v,-1)) for v in [0] + np.geomspace(30,max_time,11).tolist()] # Time points at which to make bar plots of the RNA populations

        self.plot_mean_flag = config_params['Plot parameters']['Plot mean data']
        self.best_fit_flag = config_params['Plot parameters']['Best fit']['Plot best fit']
        self.output_dir = config_params['Output location']
        # Check if "Marker shape" exists in config_params['Plot parameters']
        if "Marker shape" in config_params['Plot parameters']:
            self.marker_shape = config_params['Plot parameters']["Marker shape"]
        else:
            self.marker_shape = 'o'
        self.residual_flag = config_params['Plot parameters']['Plot residuals']
        self.RNA_populations_flag = config_params['Plot parameters']['Plot RNA population curves']
        self.RNA_populations_3D_flag = config_params['Plot parameters']['Plot 3D RNA population curves']
        self.enzyme_populations_flag = config_params['Plot parameters']['Plot enzyme population curves']
        self.annealed_fraction_flag = config_params['Plot parameters']['Plot annealed fraction']
        self.bar_2d_flag  = config_params['Plot parameters']['Plot 2D population bars']
        self.bar_3d_flag = config_params['Plot parameters']['Plot 3D population bars']
        self.xlimits = config_params['Plot parameters']['xlimits']
        self.export_ss_times = config_params['Modeling parameters']['SS time range']['Export']
        unique_enzyme = self.experiments[0].enzyme # FRET plots
        points = len(unique_enzyme)
        points = 5
        cm_input = config_params['Plot parameters']['Best fit']['Colormap']
        if type(cm_input) == str:
            colormap = cm.get_cmap(cm_input)
            trim = True
        elif type(cm_input) == list:
            N = 100
            vals = np.ones((N, 4))
            vals[:, 0] = np.linspace(cm_input[0][0]/256, cm_input[1][0]/256, N)
            vals[:, 1] = np.linspace(cm_input[0][1]/256, cm_input[1][1]/256, N)
            vals[:, 2] = np.linspace(cm_input[0][2]/256, cm_input[1][2]/256, N)
            vals[:, 3] = np.linspace(cm_input[0][3], cm_input[1][3], N)
            newcmp = colors.ListedColormap(vals, name='newcmp')

            colormap = newcmp
            trim = False
        else:
            print("Invalid colormap specified. Using inferno_r.")
            colormap = cm.inferno_r
            trim = True
        map_name = 'enzyme_colors'
        self.get_colors(points, colormap, map_name, trim)
        self.enzyme_colors = self.enzyme_colors
        self.alphas = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3]

        t_pop_keys = [k for k in kinetic_models[0].concentrations.keys() if k not in ['E', 'E*']] # 2D and 3D bar plots
        points = len(t_pop_keys)
        colormap = cm.coolwarm_r
        map_name = 't_pop_colors'
        trim = True
        self.get_colors(points, colormap, map_name, trim)

        self.enzyme_bar_colors = ['#80cdc1', '#c7eae5'] # 2D bar plots

        plot_name = config_params['Output plot file']


        self.pdf = make_pdf(f"{self.output_dir}/{plot_name}")

    def addattr(self, x, v):
        self.__dict__[x] = v

    @staticmethod
    def plot_best_fit(experiments, kinetic_models, hybridization_models, enz_colors, sample_name, pdf, best_fit_flag, plot_mean_flag, xlimits=None, marker_shape='o'):
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['mathtext.default'] = 'regular'
        if platform.system() == "Linux":
            mpl.rcParams['font.family'] = 'Arial'
        else:
            mpl.rcParams['font.family'] = 'Helvetica'

        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            kinetic_model = kinetic_models[j]
            hybridization_model = hybridization_models[j]            
            if plot_mean_flag == True:
                for r, rna in enumerate(experiment.rna):
                    for e,enzyme in enumerate(experiment.enzyme):
                        experiment.unique_time.append([])
                        experiment.mean_fret.append([])
                        experiment.fret_std.append([])                  
                        for t,time in enumerate(np.unique(experiment.time[e])):
                            filtered_fret = experiment.fret[e][np.where(experiment.time[e] == time)[0]]
                            mean_fret = filtered_fret.mean()
                            stdev_fret = filtered_fret.std()
                            stdev_fret = stdev_fret if not np.isnan(stdev_fret) else 0

                            experiment.unique_time[e].append(time)
                            experiment.mean_fret[e].append(mean_fret)
                            experiment.fret_std[e].append(stdev_fret)

            data_fit_fig = plt.subplots(1,1,figsize=(2.58,1.8)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
            data_fit_fig[0].set_linewidth(3)

            if xlimits == None:
                max_time = round(np.max([np.max(v) for v in experiment.time]),-2)
                xlim = [0-0.3 * max_time, max_time+0.3 * max_time]
            else:  
                xlim = xlimits
                max_time = xlim[1]
                xlim = [xlim[0]-0.05 * max_time, xlim[1]+0.05 * max_time]
                
            xticks = np.linspace(0,max_time,5)
            xticks = [int(x) for x in xticks]

            for i, enzyme in enumerate(experiment.enzyme):
                if float(f"{enzyme*1e6:.1f}") in [1.0, 2.0, 3.0, 5.0, 7.0]:
                    color_idx = [1, 2, 3, 5, 7].index(int(enzyme*1e6))
                elif float(f"{enzyme*1e6:.2f}") in [7.6, 15.7, 24.3, 41, 55]:
                    color_idx = [7.6, 15.7, 24.3, 41, 55].index(float(f"{enzyme*1e6:.2f}"))
                else:
                    color_idx = i
                if "Dense" in str(type(kinetic_model)):
                    line_label = f"{kinetic_model.enzyme_dense[i] * 1e6:.1f} ({kinetic_model.rna_dense[i] * 1e9:.1f} nM RNA)"
                elif "TwoPhase" in str(type(kinetic_model)):
                    line_label = f"Dense: {kinetic_model.enzyme_dense[i] * 1e6:.1f} ({kinetic_model.rna_dense[i] * 1e9:.1f} nM RNA)\nDilute: {kinetic_model.enzyme_dilute[i] * 1e6:.1f} ({kinetic_model.rna_dilute[i] * 1e9:.1f} nM RNA)"
                else:
                    line_label = f"{kinetic_model.enzyme[i] * 1e6:.1f}"


                # FRET data plots
                if plot_mean_flag == True:
                    data_fit_fig[1].errorbar(experiment.unique_time[i],experiment.mean_fret[i],yerr=experiment.fret_std[i], 
                                             fmt=marker_shape, markersize=3, mfc='w', linewidth=0.8,
                                             mec=enz_colors[color_idx], capsize=1.5, capthick=0.8,
                                             ecolor=enz_colors[color_idx], label=line_label, zorder = 3*i)
                    ylab = r'FRET'
                else:
                    data_fit_fig[1].scatter(experiment.time[i],experiment.fret[i], s=10, edgecolor=enz_colors[color_idx], facecolor="none", alpha=1, linewidth=1,label=line_label)
                    ylab = 'FRET'
                    
                if best_fit_flag == True:
                    data_fit_fig[1].plot(hybridization_model.time[i],hybridization_model.fret[i],color=enz_colors[color_idx], linewidth = 0.8, zorder = 100)
            
            for plot in [data_fit_fig]:
                for spine in ['left','bottom', 'right', 'top']:
                    plot[1].spines[spine].set_linewidth(0.6)
                plot[1].tick_params(axis='both', direction='in', width=0.6, length=2, pad=1)
                plot[1].set_xlim(xlim)
                plot[1].set_xticks(xticks)
                plot[1].set_xticklabels(xticks, rotation=-45)
                plot[1].set_xlabel('Time (s)', fontweight='bold')
                plot[1].set_ylim(-0.15, 1.15)
                plot[1].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
                plot[1].set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
                if plot == data_fit_fig:
                    plot[1].set_ylabel(ylab, fontweight='bold')
                plot[1].set_title(f"{sample_name.replace('_',' ')}")

                if plot_mean_flag == True:
                    handles, labels = plot[1].get_legend_handles_labels() # get handles and labels
                    handles = [h[0] for h in handles] # remove the errorbars
                    labels = [' '.join([label,r'$\mu$M']) for label in labels]
                    
                    L_title_word = sample_name.split('_')[0]
                    L = plot[1].legend(handles, labels, loc='upper left', bbox_to_anchor = (1.12, 1.0), 
                                       handlelength = 0.5, handletextpad = 0.2, edgecolor = 'black', 
                                       title = fr'[{L_title_word}]', title_fontsize = 8, 
                                       framealpha = 0, fontsize = 4)                                 

                if plot_mean_flag == False:
                    L = plot[1].legend(loc='upper center', title=fr"[{sample_name}] ($\mu$M)", fontsize = 4, 
                                       frameon=False, handlelength=0, handletextpad=0, markerscale=0)
                    for k,text in enumerate(L.get_texts()):
                        text.set_color(enz_colors[k])
                        # text.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground=enz_colors[k]),path_effects.Normal()])
                    for item in L.legend_handles:
                        item.set_visible(False) 

                plot[0].tight_layout()
                pdf.savefig(plot[0])
        plt.close()

    @staticmethod
    def plot_residuals(experiments, kinetic_models, residuals, enz_colors, pdf, plot_mean_flag, xlimits=None, marker_shape='o'):
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['mathtext.default'] = 'regular'

        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            residual = residuals[j]
            kinetic_model = kinetic_models[j]

            if plot_mean_flag == True:
                for r, rna in enumerate(experiment.rna):
                    for e,enzyme in enumerate(experiment.enzyme):
                        experiment.unique_time.append([])
                        experiment.mean_resid.append([])
                        experiment.resid_std.append([])                
                        for t,time in enumerate(np.unique(experiment.time[e])):
                            filtered_resid = residual[e][np.where(experiment.time[e] == time)[0]]
                            mean_resid = filtered_resid.mean()
                            stdev_resid = filtered_resid.std()
                            stdev_resid = stdev_resid if not np.isnan(stdev_resid) else 0

                            experiment.mean_resid[e].append(mean_resid)
                            experiment.resid_std[e].append(stdev_resid)

            if xlimits == None:
                max_time = round(np.max([np.max(v) for v in experiment.time]),-2)
                xlim = [0-0.05 * max_time, max_time+0.05 * max_time]
            else:  
                xlim = xlimits
                max_time = xlim[1]
                xlim = [xlim[0]-0.05 * max_time, xlim[1]+0.05 * max_time]

            xticks = np.linspace(0,max_time,5)
            xticks = [int(x) for x in xticks]

            max_resid = round(np.max([np.max(np.abs(v)) for v in residual]),2)
            ylim = [-max_resid-0.2 * max_resid, max_resid+0.2 * max_resid]
            if len(experiment.enzyme) == 1:
                fig_mod = 1
            else:
                fig_mod = 1.
            resid_fig, axs = plt.subplots(len(experiment.enzyme), 1, figsize=(3, len(experiment.enzyme) * fig_mod))
            resid_fig.suptitle(f"Fit Residuals", fontweight='bold')

            for i, enzyme in enumerate(experiment.enzyme):
                if float(f"{enzyme*1e6:.1f}") in [1.0, 2.0, 3.0, 5.0, 7.0]:
                    color_idx = [1, 2, 3, 5, 7].index(int(enzyme*1e6))
                elif float(f"{enzyme*1e6:.2f}") in [7.6, 15.7, 24.3, 41, 55]:
                    color_idx = [7.6, 15.7, 24.3, 41, 55].index(float(f"{enzyme*1e6:.2f}"))
                else:
                    color_idx = i
                if "DensePhase" in str(type(kinetic_model)):
                    line_label = f"{kinetic_model.enzyme_dense[i] * 1e6:.1f} ({kinetic_model.rna_dense[i] * 1e9:.1f} nM RNA)"
                else:
                    line_label = f"{kinetic_model.enzyme[i] * 1e6:.1f}"

                axs[i].axhline(y=0, color='k', lw=0.6, ls='--',zorder=0) # Data fit residual plots

                if plot_mean_flag == True:
                    axs[i].errorbar(experiment.unique_time[i],experiment.mean_resid[i],yerr=experiment.resid_std[i],fmt=marker_shape, markersize=3,
                                    mfc='w', mec=enz_colors[color_idx], mew=1, linewidth=0.8,
                                    capsize=1.5, capthick=0.8,ecolor=enz_colors[color_idx], label=line_label)
                else:
                    axs[i].scatter(experiment.time[i], residual[i], edgecolor=enz_colors[color_idx], facecolor="none", linewidth=1, label=line_label)
                
                if xlim[0] == xlim[1]:
                    xlim = [xlim[0]-0.05,xlim[1]+0.05]
                if ylim[0] == ylim[1]:
                    ylim = [ylim[0]-0.05,ylim[1]+0.05]
 
                axs[i].set_ylim(ylim)
                axs[i].set_xlim(xlim)
                axs[i].set_xticks(xticks) 
                axs[i].tick_params(axis='both', direction='in', width=0.6, length=2, pad=1)

                for spine in axs[i].spines.values():
                    spine.set_linewidth(0.6)
                if plot_mean_flag == True:
                    handles, labels = axs[i].get_legend_handles_labels() # get handles and labels
                    handles = [h[0] for h in handles] # remove the errorbars
                    L = axs[i].legend(handles, labels, loc='upper right', frameon=False, handlelength=0, handletextpad=0, markerscale=0)
                    for k,text in enumerate(L.get_texts()):
                        if float(f"{enzyme*1e6:.1f}") in [1.0, 2.0, 3.0, 5.0, 7.0]:
                            color_idx = [1, 2, 3, 5, 7].index(int(enzyme*1e6))
                        elif float(f"{enzyme*1e6:.2f}") in [7.6, 15.7, 24.3, 41, 55]:
                            color_idx = [7.6, 15.7, 24.3, 41, 55].index(float(f"{enzyme*1e6:.2f}"))
                        else:
                            color_idx = i
                        text.set_color(enz_colors[color_idx])
                        text.set_fontweight("bold")
                        # text.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground=enz_colors[i]),path_effects.Normal()])      
            
                if plot_mean_flag == False:
                    L = axs[i].legend(loc='upper right', frameon=False, handlelength=0, handletextpad=0, markerscale=0)
                    for k,text in enumerate(L.get_texts()):
                        text.set_color(enz_colors[i])
                        text.set_fontweight("bold")
                    for item in L.legend_handles:
                        item.set_visible(False) 

                if i < len(experiment.enzyme) - 1:
                    axs[i].set_xticklabels([])
                else:
                    axs[i].set_xticklabels(xticks, rotation=-45)
                    axs[i].set_xlabel('Time (s)', fontweight='bold')
                if i == int(np.floor(len(experiment.enzyme)/2),):
                    axs[i].set_ylabel('Fit Residuals', fontweight='bold')
                    
            resid_fig.tight_layout()
            pdf.savefig(resid_fig, bbox_inches='tight')
        plt.close()

    def plot_combined_residuals(self, experiments, kinetic_models, residuals, enz_colors, pdf, plot_mean_flag, xlimits=None, marker_shape='o'):
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['mathtext.default'] = 'regular'

        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            residual = residuals[j]
            kinetic_model = kinetic_models[j]

            if plot_mean_flag == True:
                for r, rna in enumerate(experiment.rna):
                    for e,enzyme in enumerate(experiment.enzyme):
                        experiment.unique_time.append([])
                        experiment.mean_resid.append([])
                        experiment.resid_std.append([])                
                        for t,time in enumerate(np.unique(experiment.time[e])):
                            filtered_resid = residual[e][np.where(experiment.time[e] == time)[0]]
                            mean_resid = filtered_resid.mean()
                            stdev_resid = filtered_resid.std()
                            stdev_resid = stdev_resid if not np.isnan(stdev_resid) else 0
                            experiment.unique_time[e].append(time)
                            experiment.mean_resid[e].append(mean_resid)
                            experiment.resid_std[e].append(stdev_resid)

            if xlimits == None:
                max_time = round(np.max([np.max(v) for v in experiment.time]),-2)
                xlim = [0-0.05 * max_time, max_time+0.05 * max_time]
            else:  
                xlim = xlimits
                max_time = xlim[1]
                xlim = [xlim[0]-0.05 * max_time, xlim[1]+0.05 * max_time]

            xticks = np.linspace(0,max_time,5)
            xticks = [int(x) for x in xticks]

            max_resid = round(np.max([np.max(np.abs(v)) for v in residual]), 2)
            ylim = [-max_resid-0.2 * max_resid, max_resid+0.2 * max_resid]
            if len(experiment.enzyme) == 1:
                fig_mod = 1
            else:
                fig_mod = 1.

            resid_fig, axs = plt.subplots(1, 1, figsize=(2.1, 1.2))
            resid_fig.suptitle(f"Fit Residuals", fontweight='bold')
            axs.axhline(y=0, color='k',lw=0.6,ls='--',zorder=0) # Data fit residual plots

            for i, enzyme in enumerate(experiment.enzyme):
                if float(f"{enzyme*1e6:.1f}") in [1.0, 2.0, 3.0, 5.0, 7.0]:
                    color_idx = [1, 2, 3, 5, 7].index(int(enzyme*1e6))
                elif float(f"{enzyme*1e6:.2f}") in [7.6, 15.7, 24.3, 41, 55]:
                    color_idx = [7.6, 15.7, 24.3, 41, 55].index(float(f"{enzyme*1e6:.2f}"))
                else:
                    color_idx = i
                if "DensePhase" in str(type(kinetic_model)):
                    line_label = f"{kinetic_model.enzyme_dense[i] * 1e6:.1f} ({kinetic_model.rna_dense[i] * 1e9:.1f} nM RNA)"
                else:
                    line_label = f"{kinetic_model.enzyme[i] * 1e6:.1f}"


                if plot_mean_flag == True:
                    axs.errorbar(experiment.unique_time[i], experiment.mean_resid[i], yerr=experiment.resid_std[i], 
                                 fmt=marker_shape, markersize=2, mfc='w', mec=enz_colors[color_idx], linewidth=0.6,
                                 capsize=1.5, capthick=0.6, ecolor=enz_colors[color_idx], label=line_label, zorder = 10*len(experiment.enzyme)-10*i)
                else:
                    axs.scatter(experiment.time[i], residual[i], color=enz_colors[color_idx], alpha=0.8, linewidth=0, label=line_label)
                
                if xlim[0] == xlim[1]:
                    xlim = [xlim[0]-0.05,xlim[1]+0.05]
                if ylim[0] == ylim[1]:
                    ylim = [ylim[0]-0.05,ylim[1]+0.05]
 
                axs.set_ylim(ylim)
                axs.set_xlim(xlim)
                axs.set_xticks(xticks) 
                axs.tick_params(axis='both', direction='in', width=0.6, length=2, pad=1)
                
                for spine in axs.spines.values():
                    spine.set_linewidth(0.6)
                
                if plot_mean_flag == False:
                    L = axs.legend(loc='upper right',frameon=False,handlelength=0,handletextpad=0,markerscale=0)
                    for k,text in enumerate(L.get_texts()):
                        text.set_color(enz_colors[i])
                        text.set_fontweight("bold")
                    for item in L.legend_handles:
                        item.set_visible(False) 

            if plot_mean_flag == True:
                handles, labels = axs.get_legend_handles_labels() # get handles and labels
                handles = [h[0] for h in handles] # remove the errorbars
                L = axs.legend(handles, labels, loc='upper left', frameon=False, handlelength=0, 
                               handletextpad=0, markerscale=0, bbox_to_anchor=(1.03, 1.0))
                for k,text in enumerate(L.get_texts()):
                    if float(text.get_text()) in [1.0, 2.0, 3.0, 5.0, 7.0]:
                        color_idx = [1, 2, 3, 5, 7].index(float(text.get_text()))
                    elif float(text.get_text()) in [7.6, 15.7, 24.3, 41, 55]:
                        color_idx = [7.6, 15.7, 24.3, 41, 55].index(float(text.get_text()))
                    else:
                        color_idx = k
                    text.set_color(enz_colors[color_idx])
                    text.set_fontweight("bold")
        
            axs.set_xticklabels(xticks, rotation=-45)
            axs.set_xlabel('Time (s)', fontweight='bold')
            axs.set_ylabel('Fit Residuals', fontweight='bold')
                    
            resid_fig.tight_layout()
            pdf.savefig(resid_fig, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_rna_populations(experiments, kinetic_models, sample_name, pdf, output_dir, export_ss_times=False, xlimits=None):
        mpl.rcParams['mathtext.default'] = 'regular'  
        plt.rcParams['figure.constrained_layout.use'] = True 
        mpl.rcParams['font.size'] = 8

        all_ss_times = [] 
         
        if xlimits == None:
            max_time = round(np.max([np.max(v) for v in experiment.time]),-2)
            xlim = [0-0.05 * max_time, max_time+0.05 * max_time]
        else:  
            xlim = xlimits
            max_time = xlim[1]
            xlim = [xlim[0]-0.05 * max_time, xlim[1]+0.05 * max_time]
        
        for j, experiment in enumerate(experiments): # Only make plot for one replicate
            if j == 0:
                kinetic_model = kinetic_models[j]
                kinetic_model.calculate_total_rna_concentrations()
                color_values = cm.coolwarm(np.linspace(0,1,len(kinetic_model.total_rna_concentrations.keys())))
                if len(experiment.enzyme) == 1:
                    fig_mod = 5.2
                else:
                    fig_mod = 1.7
                rna_pop_fig = plt.figure(figsize=(6, len(experiment.enzyme) * fig_mod))
                rna_pop_fig.suptitle(f"RNA populations for {sample_name}", fontsize=16, fontweight='bold')

                if len(experiment.enzyme) == 1:
                    apf_subfigs = rna_pop_fig.subfigures(2,1)
                else:
                    apf_subfigs = rna_pop_fig.subfigures(len(experiment.enzyme),1)

                for i, enzyme in enumerate(experiment.enzyme):                    
                    axs = apf_subfigs[i].subplots(1,5,gridspec_kw={'width_ratios': [2,0.05,0.001,1,1]})
                    enzyme = experiment.enzyme[i]
                    population_sum = np.zeros(len(kinetic_model.time[i]))
                    apf_subfigs[i].suptitle(fr"{enzyme * 1e6} $\mu$M enzyme", fontweight='bold')
                    
                    # Individual RNA populations plots
                    for q, k in enumerate(kinetic_model.total_rna_concentrations.keys()):
                        # if k == "TA2":
                        #     ic(kinetic_model.time[i], kinetic_model.total_rna_concentrations[k][i])
                        if k != 'A1': 
                            axs[0].plot(kinetic_model.time[i], kinetic_model.total_rna_concentrations[k][i], label=k, color=color_values[q], alpha=0.8)                
                            if k!= 'TA1':
                                population_sum += kinetic_model.total_rna_concentrations[k][i]
                        else: # Plot A1 on separate axis since it gets very large
                            axs[4].plot(kinetic_model.time[i], kinetic_model.total_rna_concentrations[k][i], label=k, color='black', alpha=0.8)
                    
                    # Total cleavable RNA plot
                    max_time = max(kinetic_model.time[i])
                    round_value = int(str(kinetic_model.rna[0]).split('0')[-1]) + 2
                    axs[3].plot(kinetic_model.time[i], population_sum, label='Total RNA', color='black', alpha=0.8)
                    try:
                        ss_index = max([i for i, x in enumerate(population_sum) if np.round(x,round_value) == kinetic_model.rna[0]])
                        ss_time = kinetic_model.time[i][ss_index]
                    except:
                        ss_time = None
                    if ss_time is not None:
                        all_ss_times.append([enzyme, ss_time])
                        text_time = ss_time + 100 if ss_time < max_time / 2 else ss_time - 100
                        direction = 'left' if ss_time < max_time / 2 else 'right'
                        axs[3].axvline(x=ss_time, color='red', linestyle='--', alpha=0.8)
                        axs[3].text(text_time, 0, f"{int(ss_time)} s", color='red', fontsize=10, ha=direction, va='bottom',
                                    bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
                    
                    # Labels and formatting
                    axs[2].axis('off')
                    rna_modifier = kinetic_model.rna[0] * 0.05
                    titles = ['All RNA species', 'Total Cleavable RNA', '[A1]']
                    ylabeltext = ['Concentration (M)', 'Concentration (M)', 'Concentration (M)']
                    ylimits = [[0 - rna_modifier, kinetic_model.rna[0] + rna_modifier],
                            [0 - rna_modifier, kinetic_model.rna[0] + rna_modifier],
                            [0 - 10 * rna_modifier, kinetic_model.rna[0] * experiment.n + 10 * rna_modifier],
                            [-0.05, 1.05]] 
                    for ani, ax_num in enumerate([0, 3, 4]):
                        if i == 0:
                            axs[ax_num].set_title(titles[ani],fontweight='bold')
                        if i == len(experiment.enzyme) - 1:  # Put x-axis label below last plot
                            axs[ax_num].set_xlabel('Time (s)', fontweight='bold')
                        axs[ax_num].set_ylabel(ylabeltext[ani], fontweight='bold')
                        axs[ax_num].ticklabel_format(axis='both', style='sci', scilimits=[-2,2], useMathText=True)
                        axs[ax_num].set_ylim(ylimits[ani])
                        axs[ax_num].set_xlim(xlim)

                    cbar = rna_pop_fig.colorbar(cm.ScalarMappable(cmap='coolwarm'), cax=axs[1], label="Length polyA tail", ticks=[0, 0.33, 0.66, 1], location='left')
                    cbar.set_ticklabels([str(1), str(int(experiment.n/3)), str(int(2 * experiment.n/3)), str(experiment.n)])
                if len(experiment.enzyme) == 1:
                    apf_subfigs[1].set_visible(False)
                pdf.savefig(rna_pop_fig,bbox_inches='tight')
            plt.close()
            
            if "2phase" in kinetic_model.overall_model:
                for i, enzyme in enumerate(experiment.enzyme): 
                    maxy = -1000
                    miny = 1000
                    maxy_a1 = -1000
                    miny_a1 = 1000
                    color_values = cm.coolwarm(np.linspace(0,1,len(kinetic_model.total_rna_concentrations.keys())))
                    enz_by_phase = [kinetic_model.enzyme_dense[i], kinetic_model.enzyme_dilute[i]]
                    rna_by_phase = [kinetic_model.rna_dense[i], kinetic_model.rna_dilute[i]]
                    fig, axs = plt.subplots(2, 4, figsize = (6,4), gridspec_kw={'width_ratios': [1, 0.05, 0.7, 0.7]})
                    for pi, phase in enumerate(['Dense', 'Dilute']):
                        all_species = [0]*len(kinetic_model.time[i])
                        num_rna_species = len(kinetic_model.separate_rnas[phase])-1
                        for si, species_data in enumerate(kinetic_model.separate_rnas[phase]):
                            if si == 0:
                                axs[pi, 3].plot(kinetic_model.time[i], species_data, color = 'black', zorder = 2 * num_rna_species - si, label='A1')
                                maxy_a1 = max(maxy_a1, max(species_data))
                                miny_a1 = min(miny_a1, min(species_data))
                            else:
                                species_data = kinetic_model.separate_rnas[phase][si]
                                axs[pi, 0].plot(kinetic_model.time[i], species_data, color = color_values[si], zorder = 2 * num_rna_species - si)
                                if si > 1:
                                    all_species = [x+y for x,y in zip(all_species, species_data)]
                                
                                maxy = max(maxy, max(species_data))
                                miny = min(miny, min(species_data))
                        axs[pi,0].set_title(f"{phase} phase:\n{enz_by_phase[pi] * 1e6:.1f} uM enzyme\n{rna_by_phase[pi] * 1e9:.1f} nM RNA")            
                        cbar = fig.colorbar(cm.ScalarMappable(cmap='coolwarm'), cax=axs[pi, 1], label="Length polyA tail", ticks=[0, 0.33, 0.66, 1], location='left')
                        
                        cbar.set_ticklabels([str(1), str(int(num_rna_species/3)), str(int(2 * num_rna_species/3)), str(num_rna_species)])
                        axs[pi, 2].plot(kinetic_model.time[i], all_species, color='black', linestyle='--')
                        axs[pi, 2].set_title("\n".join(["[Total cleavable RNA]", f"in {phase} phase"]))
                        axs[pi, 3].set_title("\n".join(["[A1]", f"in {phase} phase"]))
                    
                    # all_species = [0]*len(kinetic_model.time[i])
                    # for si, species in enumerate(kinetic_model.total_rna_concentrations.keys()):
                    #     tmp_data = kinetic_model.total_rna_concentrations[species][0]
                    #     if species == 'A1':
                    #         axs[2, 3].plot(kinetic_model.time[i], tmp_data, color = 'black', zorder = 2 * num_rna_species - si, label='Total RNA')
                    #         maxy_a1 = max(maxy_a1, max(tmp_data))
                    #         miny_a1 = min(miny_a1, min(tmp_data))
                    #     else:
                    #         axs[2, 0].plot(kinetic_model.time[i], tmp_data, color = color_values[si], zorder = 2 * num_rna_species - si)
                    #         if si > 1:
                    #             all_species = [x+y for x,y in zip(all_species, tmp_data)]
                    #         maxy = max(maxy, max(tmp_data))
                    #         miny = min(miny, min(tmp_data))
                    # axs[2,0].set_title(f"All RNA")            
                    # cbar = fig.colorbar(cm.ScalarMappable(cmap='coolwarm'), cax=axs[2, 1], label="Length polyA tail", ticks=[0, 0.33, 0.66, 1], location='left')
                    # cbar.set_ticklabels([str(1), str(int(num_rna_species/3)), str(int(2 * num_rna_species/3)), str(num_rna_species)])
                    # axs[2, 2].plot(kinetic_model.time[i], all_species, color='black', linestyle='--')
                    # axs[2, 2].set_title(f"[Total cleavable RNA]")   
                    # axs[2, 3].set_title(f"[A1]")    
                    for i in range(2):
                        range_y = maxy - miny
                        axs[i,0].set_ylim([miny - 0.1 * range_y, maxy + 0.1 * range_y])
                        axs[i,2].set_ylim([miny - 0.1 * range_y, maxy + 0.1 * range_y])

                        range_y_a1 = maxy_a1 - miny_a1
                        axs[i,3].set_ylim([miny_a1 - 0.1 * range_y_a1, maxy_a1 + 0.1 * range_y_a1])

                    fig.supxlabel("Time (s)")
                    fig.supylabel(r"[RNA(A)$_i$] (M)")

                    fig.suptitle(f"[Enzyme] = {enzyme * 1e6} uM")

                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

            if export_ss_times == True:
                np.savetxt(f"{output_dir}/{sample_name}_ss_times.csv", all_ss_times, delimiter=",", header="Enzyme,SteadyStateTime", comments="")

    @staticmethod
    def plot_rna_populations_3D(experiments, kinetic_models, sample_name, pdf, enz_colors, xlimits=None):
        mpl.rcParams['mathtext.default'] = 'regular'  
        plt.rcParams['figure.constrained_layout.use'] = True 
        mpl.rcParams['font.size'] = 12
         
        if xlimits == None:
            max_time = round(np.max([np.max(v) for v in experiment.time]),-2)
            xlim = [0-0.05 * max_time, max_time+0.05 * max_time]
        else:  
            xlim = xlimits
            max_time = xlim[1]
            xlim = [xlim[0]-0.05 * max_time, xlim[1]+0.05 * max_time]
        
        for j, experiment in enumerate(experiments): # Only make plot for one replicate
            if j == 0:
                kinetic_model = kinetic_models[j]
                kinetic_model.calculate_total_rna_concentrations()
                color_values = cm.coolwarm(np.linspace(0,1,len(kinetic_model.total_rna_concentrations.keys())))

                rna_pop_fig_3d = plt.figure(figsize=(6,6))
                rna_pop_fig_3d.suptitle(f"RNA populations for {sample_name} 1/2", fontsize=16, fontweight='bold')
                
                rna_pop_fig_2d = plt.figure(figsize=(6, 3))
                rna_pop_fig_2d.suptitle(f"RNA populations for {sample_name} 2/2", fontsize=16, fontweight='bold')

                ax0 = rna_pop_fig_3d.add_subplot(projection='3d')  # Create a subplot in the first grid cell
                
                axs = gridspec.GridSpec(1, 3, figure=rna_pop_fig_2d, width_ratios=[1, 1, 0.1])  # Create a grid for the 2D plots
                
                ax3 = rna_pop_fig_2d.add_subplot(axs[0])  # Create a subplot in the fourth grid cell
                ax4 = rna_pop_fig_2d.add_subplot(axs[1])
                ax5 = rna_pop_fig_2d.add_subplot(axs[2])
                max_enz = max(experiment.enzyme)

                for i, enzyme in enumerate(experiment.enzyme):                    
                    population_sum = np.zeros(len(kinetic_model.time[i]))
                    tot_keys = len(kinetic_model.total_rna_concentrations.keys())
                    fill_between_array = np.zeros(len(kinetic_model.time[i]))
                    # Individual RNA populations plots
                    for q, k in enumerate(kinetic_model.total_rna_concentrations.keys()):
                        if k != 'A1': 
                            ax0.plot(kinetic_model.time[i], [i] * len(kinetic_model.time[i]), kinetic_model.total_rna_concentrations[k][i], 
                                     label=k, color=color_values[q], linewidth=1.5,
                                     zorder = (tot_keys+1) * (max_enz - i) - q )
                                    # Find the largest RNA population at each time point and highlight it
                            if k!= 'TA1':
                                population_sum += kinetic_model.total_rna_concentrations[k][i]
                                fill_between_array = np.maximum(fill_between_array, kinetic_model.total_rna_concentrations[k][i])
                        else: # Plot A1 on separate axis since it gets very large
                            ax4.plot(kinetic_model.time[i], kinetic_model.total_rna_concentrations[k][i], label=k, color=enz_colors[i], alpha=0.8)
                    ax0.set_yticks(np.arange(len(experiment.enzyme)))
                    ax0.set_yticklabels([e * 1e6 for e in experiment.enzyme])  # Hide y-tick labels for the first plot
                    # if enzyme != 0:
                    #     ax0.fill_between(list(kinetic_model.time[i]), [i] * len(kinetic_model.time[i]), [0] * len(kinetic_model.time[i]), list(kinetic_model.time[i]), [i] * len(kinetic_model.time[i]), fill_between_array) # Fill between the largest RNA population and 0//facecolors="white", alpha=1, zorder=(tot_keys+1)*(max_enz - i) - q - 1
                    # Total cleavable RNA plot
                    max_time = max(kinetic_model.time[i])
                    ax3.plot(kinetic_model.time[i], population_sum,  label='Total RNA', color=enz_colors[i], alpha=0.8)

                ax0.view_init(elev=20, azim=-60)  # Set the elevation and azimuthal angles for the first plot

                # Labels and formatting
                rna_modifier = kinetic_model.rna[0] * 0.05
                titles = ['All RNA species', 'Total Cleavable RNA', '[A1]']
                ylabeltext = ['Concentration (M)', 'Concentration (M)', 'Concentration (M)']
                ylimits = [[0 - rna_modifier, kinetic_model.rna[0] + rna_modifier],
                        [0 - rna_modifier, kinetic_model.rna[0] + rna_modifier],
                        [0 - 10 * rna_modifier, kinetic_model.rna[0] * experiment.n + 10 * rna_modifier],
                        [-0.05, 1.05]] 
                for ani, ax in enumerate([ax0, ax3, ax4]):
                    ax.set_title(titles[ani],fontweight='bold')
                    ax.set_xlabel('Time (s)', fontweight='bold')
                    if ani == 0:
                        ax.set_ylabel(r'Enzyme Concentration ($\mu$M)', fontweight='bold')
                        ax.set_zlabel(ylabeltext[ani], fontweight='bold')
                        ax.set_zlim3d([0,ylimits[ani][1]])
                        ax.set_xlim3d([0,xlim[1]])
                    else:
                        ax.set_ylabel(ylabeltext[ani], fontweight='bold')
                        ax.ticklabel_format(axis='both', style='sci', scilimits=[-2,2], useMathText=True)
                        ax.set_ylim(ylimits[ani])
                        ax.set_xlim(xlim)
                
                # Add a list of enzyme concentrations colored by their respective enzyme color in ax5
                for idx, enzyme in enumerate(experiment.enzyme):
                    ax5.text(
                        0.3, 
                        1 - idx * 0.12, 
                        f"{enzyme * 1e6:.2f} μM", 
                        color=enz_colors[idx], 
                        fontsize=10, 
                        transform=ax5.transAxes, 
                        va='top', ha = 'center',
                        path_effects=[path_effects.Stroke(linewidth=1.2, foreground=enz_colors[idx]), path_effects.Normal()]
                    )
                ax5.set_title("\n".join(["Enzyme","concentrations"]), fontweight='bold')
                ax5.axis('off')
                
                rna_pop_fig_3d.tight_layout()
                pdf.savefig(rna_pop_fig_3d)
                pdf.savefig(rna_pop_fig_2d, bbox_inches='tight')
            plt.close()
            
    @staticmethod
    def plot_enzyme_populations(experiments, kinetic_models, sample_name, pdf, xlimits=None):
        mpl.rcParams['mathtext.default'] = 'regular'  
        plt.rcParams['figure.constrained_layout.use'] = True 
        mpl.rcParams['font.size'] = 12

        if xlimits == None:
            max_time = round(np.max([np.max(v) for v in experiment.time]),-2)
            xlim = [0-0.05 * max_time, max_time+0.05 * max_time]
        else:  
            xlim = xlimits
            max_time = xlim[1]
            xlim = [xlim[0]-0.05 * max_time, xlim[1]+0.05 * max_time]

        for j, experiment in enumerate(experiments): # Only make plot for one replicate
            if j == 0:
                kinetic_model = kinetic_models[j]
                kinetic_model.calculate_total_rna_concentrations()


                non_zero_enz = [enz for enz in experiment.enzyme if float(enz) != 0.]
                non_zero_idx = [ei for ei, enz in enumerate(experiment.enzyme) if float(enz) != 0.]

                if len(non_zero_enz) == 1:
                    fig_mod = 5.2
                else:
                    fig_mod = 2.2

                enz_pop_fig = plt.figure(figsize=(9, len(non_zero_enz) * fig_mod))
                enz_pop_fig.suptitle(f"Enzyme species for {sample_name}", fontsize=16, fontweight='bold')
                

                if len(non_zero_enz) == 1:
                    apf_subfigs = enz_pop_fig.subfigures(2,1)
                else:
                    apf_subfigs = enz_pop_fig.subfigures(len(non_zero_enz),1)
                
                for i, enz in enumerate(non_zero_enz):
                    axs = apf_subfigs[i].subplots(1,3,gridspec_kw={'width_ratios': [1,1,1]})
                    ei = non_zero_idx[i]
                    enzyme = experiment.enzyme[ei]
                    apf_subfigs[i].suptitle(fr"{enzyme * 1e6} $\mu$M enzyme", fontweight='bold')

                    # All enzyme populations plot
                    enz_style = ['--', ':', '-']
                    
                    sum_E, sum_ETA = np.zeros(len(kinetic_model.time[ei])), np.zeros(len(kinetic_model.time[ei]))
                    for key in kinetic_model.concentrations.keys():
                        if 'ETA' in key:
                            sum_ETA += np.array(kinetic_model.concentrations[key][ei])
                        if 'E' in key:
                            sum_E += np.array(kinetic_model.concentrations[key][ei])

                    for b, c in enumerate(['E','E*']):
                        axs[0].plot(kinetic_model.time[ei], [x/y for x,y in zip(kinetic_model.concentrations[c][ei],sum_E)], label=c, color='black', linestyle=enz_style[b])
                    axs[0].plot(kinetic_model.time[ei], [x/y for x,y in zip(sum_ETA,sum_E)], label=r'$ETA_{i}$', color='black', linestyle=enz_style[2])
                    axs[0].legend(frameon=True,loc='upper right', facecolor='white')
                    axs[0].set_ylim([-0.05, 1.05])

                    # Free enzyme populations plot
                    enz_style = ['-', ':']
                    sum_E = [x+y for x,y in zip(kinetic_model.concentrations['E'][ei], kinetic_model.concentrations['E*'][ei])]
                    for b, c in enumerate(['E','E*']):
                        axs[1].plot(kinetic_model.time[ei], [x/y for x,y in zip(kinetic_model.concentrations[c][ei],sum_E)], label=c, color='black', linestyle=enz_style[b])
                    axs[1].legend(frameon=True, loc='upper right', facecolor='white')
                    axs[1].set_ylim([-0.05, 1.05])

                    # E*/E plot
                    Estar_E = [x/y for x,y in zip(kinetic_model.concentrations['E*'][ei],kinetic_model.concentrations['E'][ei])]
                    axs[2].plot(kinetic_model.time[ei], Estar_E, label=c, color='black', linestyle='-')
                    temp_ylimits = axs[2].get_ylim()
                    if temp_ylimits[1] - temp_ylimits[0] < 10:
                        axs[2].set_yscale('linear')
                    else:
                        axs[2].set_yscale('log')
                    temp_ylimits = axs[2].get_ylim()
                    axs[2].set_ylim([temp_ylimits[0] - temp_ylimits[1] * 0.1, temp_ylimits[1] + temp_ylimits[1] * 0.1])
                    if axs[2].get_ylim()[1] - axs[2].get_ylim()[0] < 1:
                        axs[2].set_ylim([axs[2].get_ylim()[0]-0.5, axs[2].get_ylim()[1]+0.5])
                    axs[2].annotate(fr"E*/E ~ {sci_notation(np.mean(Estar_E))} $\pm$ {sci_notation(np.std(Estar_E))}", 
                                    (0.05,0.9), xycoords='axes fraction', fontsize=8, ha='left', color='black',
                                bbox=dict(fc="w", ec='none', alpha=0.8, pad=0.5))

                    # Labels and formatting
                    titles = ['Fraction All Enzyme', 'Fraction Free Enzyme', 'E*/E Ratio']
                    ylabeltext = ['Fraction All Enzyme', 'Fraction Free Enzyme', 'E*/E']
                    for ani, ax_num in enumerate(axs):
                        axs[ani].set_xlim(xlim)
                        if i == 0:
                            axs[ani].set_title(titles[ani],fontweight='bold')
                        if i == len(experiment.enzyme) - 1:  # Put x-axis label below last plot
                            axs[ani].set_xlabel('Time (s)', fontweight='bold')
                        axs[ani].set_ylabel(ylabeltext[ani], fontweight='bold')
                        if ani != 2:
                            axis_format = 'both'
                        else:
                            axis_format = 'x'
                        axs[ani].ticklabel_format(axis=axis_format, style='sci', scilimits=[-2,2], useMathText=True)
                if len(experiment.enzyme) == 1:
                    apf_subfigs[1].set_visible(False)

                pdf.savefig(enz_pop_fig,bbox_inches='tight')

            else:
                for i, subfig in enumerate(apf_subfigs): # for each enzyme concentration
                    i = i + 1
                    axs = subfig.subplots(1,3,gridspec_kw={'width_ratios': [1,1,1]})
                    enzyme = experiment.enzyme[i]
                    subfig.suptitle(fr"{enzyme * 1e6} $\mu$M enzyme", fontweight='bold')

                    # All enzyme populations plot
                    enz_style = ['--', ':', '-']
                    
                    sum_E, sum_ETA = np.zeros(len(kinetic_model.time[i])), np.zeros(len(kinetic_model.time[i]))
                    for key in kinetic_model.concentrations.keys():
                        if 'ETA' in key:
                            sum_ETA += np.array(kinetic_model.concentrations[key][i])
                        if 'E' in key:
                            sum_E += np.array(kinetic_model.concentrations[key][i])

                    for b, c in enumerate(['E','E*']):
                    
                        axs[0].plot(kinetic_model.time[i], [x/y for x,y in zip(kinetic_model.concentrations[c][i],sum_E)], label=c, color='black', linestyle=enz_style[b])
                    axs[0].plot(kinetic_model.time[i], [x/y for x,y in zip(sum_ETA,sum_E)], label=r'$ETA_{i}$', color='black', linestyle=enz_style[2])
                    axs[0].legend(frameon=True,loc='upper right', facecolor='white')
                    axs[0].set_ylim([-0.05, 1.05])

                    # Free enzyme populations plot
                    enz_style = ['-', ':']
                    sum_E = [x+y for x,y in zip(kinetic_model.concentrations['E'][i], kinetic_model.concentrations['E*'][i])]
                    for b, c in enumerate(['E','E*']):
                        axs[1].plot(kinetic_model.time[i], [x/y for x,y in zip(kinetic_model.concentrations[c][i],sum_E)], label=c, color='black', linestyle=enz_style[b])
                    axs[1].legend(frameon=True, loc='upper right', facecolor='white')
                    axs[1].set_ylim([-0.05, 1.05])

                    # E*/E plot
                    Estar_E = [x/y for x,y in zip(kinetic_model.concentrations['E*'][i],kinetic_model.concentrations['E'][i])]
                    axs[2].plot(kinetic_model.time[i], Estar_E, label=c, color='black', linestyle='-')
                    temp_ylimits = axs[2].get_ylim()
                    if temp_ylimits[1] - temp_ylimits[0] < 10:
                        axs[2].set_yscale('linear')
                    else:
                        axs[2].set_yscale('log')
                    temp_ylimits = axs[2].get_ylim()
                    axs[2].set_ylim([temp_ylimits[0] - temp_ylimits[1] * 0.1, temp_ylimits[1] + temp_ylimits[1] * 0.1])
                    if axs[2].get_ylim()[1] - axs[2].get_ylim()[0] < 1:
                        axs[2].set_ylim([axs[2].get_ylim()[0]-0.5, axs[2].get_ylim()[1]+0.5])
                    axs[2].annotate(fr"E*/E ~ {sci_notation(np.mean(Estar_E))} $\pm$ {sci_notation(np.std(Estar_E))}", 
                                    (0.05,0.9), xycoords='axes fraction', fontsize=8, ha='left', color='black',
                                bbox=dict(fc="w", ec='none', alpha=0.8, pad=0.5))

                    # Labels and formatting
                    titles = ['Fraction All Enzyme', 'Fraction Free Enzyme', 'E*/E Ratio']
                    ylabeltext = ['Fraction All Enzyme', 'Fraction Free Enzyme', 'E*/E']
                    for ani, ax_num in enumerate(axs):
                        axs[ani].set_xlim(xlim)
                        if i == 1:
                            axs[ani].set_title(titles[ani],fontweight='bold')
                        if i == len(experiment.enzyme) - 1:  # Put x-axis label below last plot
                            axs[ani].set_xlabel('Time (s)', fontweight='bold')
                        axs[ani].set_ylabel(ylabeltext[ani], fontweight='bold')
                        if ani != 2:
                            axis_format = 'both'
                        else:
                            axis_format = 'x'
                        axs[ani].ticklabel_format(axis=axis_format, style='sci', scilimits=[-2,2], useMathText=True)

                pdf.savefig(enz_pop_fig,bbox_inches='tight')   
            plt.close()

    @staticmethod
    def plot_annealed_fraction(experiments, hybridization_models, enz_colors, sample_name, pdf):
        mpl.rcParams['mathtext.default'] = 'regular'
        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            if j == 0:
                hybridization_model = hybridization_models[j]
                annealed_fraction_fig = plt.subplots(1,1,figsize=(7,5)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]

                for i, enzyme in enumerate(experiment.enzyme):
                    color_idx = i
                    line_label = f"{enzyme * 1e6}"

                    # Annealed Fraction plot
                    annealed_fraction_fig[1].scatter(hybridization_model.time[i],hybridization_model.annealed_fraction[i],s=30,color=enz_colors[color_idx],label=line_label,alpha=0.8,linewidth=0)
                
                annealed_fraction_fig[1].set_xlabel('Time (s)', fontweight='bold')
                annealed_fraction_fig[1].set_ylabel('Annealed fraction', fontweight='bold')
                annealed_fraction_fig[1].set_title(f"Annealed fraction", fontweight='bold')
                annealed_fraction_fig[1].set_ylim([-0.1, 1.1])
                L = annealed_fraction_fig[1].legend(frameon=False,handlelength=0,handletextpad=0,loc='upper right',title=fr"[{sample_name}] $\mu$M",markerscale=0)
                for k,text in enumerate(L.get_texts()):
                    text.set_color(enz_colors[k])
                    text.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground=enz_colors[k]),path_effects.Normal()])
                annealed_fraction_fig[0].tight_layout()
                pdf.savefig(annealed_fraction_fig[0])
            plt.close()

    @staticmethod
    def plot_2d_population_bars(experiments, kinetic_models, enzyme_colors, t_pop_colors, pdf, timesample):
        mpl.rcParams['mathtext.default'] = 'regular'
        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            if j == 0:
                kinetic_model = kinetic_models[j]

                for i, enzyme in enumerate(experiment.enzyme):
                    # Species concentration bar plots at desired time points
                    fig, ax = plt.subplots(len(timesample), 3, figsize=(11, len(timesample) * 2), gridspec_kw={'width_ratios': [1, 1, 6]})

                    for ti, time in enumerate(timesample):
                        tindex = (np.abs(kinetic_model.time[i] - time)).argmin()
                        kinetic_model.calculate_total_rna_concentrations()
                        ax[ti][0].bar(0.0, kinetic_model.concentrations['E*'][i][tindex]/kinetic_model.enzyme[i], color=enzyme_colors[1], label='E*', width=0.5)
                        ax[ti][0].bar(0.75, kinetic_model.concentrations['E'][i][tindex]/kinetic_model.enzyme[i], color=enzyme_colors[0], label='E', width=0.5)
                        ax[ti][1].bar(0.0, kinetic_model.total_rna_concentrations[f'TA{kinetic_model.n}'][i][tindex]/kinetic_model.rna, color=t_pop_colors[-1], label=f"TA$_{{{kinetic_model.n}}}$", width=0.5)
                        ax[ti][1].bar(0.75, kinetic_model.total_rna_concentrations['A1'][i][tindex]/(kinetic_model.rna * (kinetic_model.n - 1)), color=t_pop_colors[0], label="A$_{{{1}}}$", width=0.5)

                        for q, k in enumerate(kinetic_model.total_rna_concentrations.keys()):
                            if k not in ['A1', f"TA{kinetic_model.n}"]:
                                alen = int(k.split('TA')[1])
                                ax[ti][2].bar(q, kinetic_model.total_rna_concentrations[k][i][tindex]/kinetic_model.rna, color=t_pop_colors[q], label=f'TA$_{{{alen}}}$')
                        
                        ax[ti][2].invert_xaxis()
                        ax[ti][0].set_ylabel(f"Fraction at t: {np.round(kinetic_model.time[i][tindex],0)} s", fontweight='bold')
                        xlabeltext = ['Enzyme states', 'Initial RNA and AMP', 'Product RNA']
                        xticktext = [['E*', 'E'], [f"TA$_{{{kinetic_model.n}}}$", 'A$_{{{1}}}$'], ['TA' + f"$_{{{v}}}$" for v in rna_lens]]
                        for ani, ax_num in enumerate([0, 1, 2]):
                            if ti == len(timesample) - 1:
                                ax[ti][ax_num].set_xlabel(xlabeltext[ani], fontweight='bold')
                            if ti < len(timesample) - 1:
                                ax[ti][ax_num].set_xticklabels([])
                            else:
                                ax[ti][ax_num].set_xticklabels(xticktext[ani], rotation=45)
                            if ax_num == 2:
                                xtickvalues = [x for x in range(0, kinetic_model.n - 1)]
                                xlimit = [kinetic_model.n-0.75, -0.75]
                            else:
                                xtickvalues = [0, 0.75]
                                xlimit = [-0.5, 1.25]
                            ax[ti][ax_num].set_xticks(xtickvalues)
                            ax[ti][ax_num].set_xlim(xlimit)
                            ax[ti][ax_num].set_ylim([-0.02, 1.02])
                        
                        rna_lens = [int(k.split('TA')[1]) for k in kinetic_model.total_rna_concentrations.keys() if k not in [f'TA{kinetic_model.n}', 'A1']]

                        if ti == 0:
                            ax[ti][2].set_title(fr"$E_{0}$: {np.round(kinetic_model.enzyme[i] * 1e6,1)}, $RNA_{0}$: {np.round(kinetic_model.rna * 1e6,1)} $\mu$M")
                    
                    fig.tight_layout()
                    pdf.savefig(fig)
        plt.close()

    @staticmethod
    def plot_3d_population_bars(experiments, kinetic_models, hybridization_models, pdf, timesample, sample_name):
        mpl.rcParams['mathtext.default'] = 'regular'
        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            kinetic_model = kinetic_models[j]
            kinetic_model.calculate_total_rna_concentrations()
            hybridization_model = hybridization_models[j]

            # 3D surface/bar plots, e.g. x = species, y = enzyme conc., z = species fraction
            surf_figs = []
            for ti, time in enumerate(timesample):
                species_matrix = []
                resampled_species_matrix = []
                enzyme_matrix = []
                resampled_enzyme_matrix = []
                fraction_matrix = []
                rna_species = [f'TA{x}' for x in range(1, experiment.n + 1)]
                fine_enz = np.linspace(kinetic_model.enzyme[1] * 1e6, kinetic_model.enzyme[-1] * 1e6, 10)
                resampled_species_vector = np.linspace(1, kinetic_model.n, kinetic_model.n)
                for ei, enz in enumerate(kinetic_model.enzyme):
                    if ei == 0:
                        pass
                    else:
                        tindex = (np.abs(kinetic_model.time[ei] - time)).argmin()
                        species_vector = [int(x.split('TA')[1]) for x in rna_species]
                        enzyme_vector = [enz * 1e6 for x in species_vector]
                        fraction_vector = [kinetic_model.total_rna_concentrations[x][ei][tindex]/kinetic_model.rna for x in rna_species]

                        species_matrix.append(species_vector)
                        enzyme_matrix.append(enzyme_vector)
                        fraction_matrix.append(fraction_vector)
                
                for ei, enz in enumerate(fine_enz):
                    resampled_enzyme_vector = [enz for x in resampled_species_vector]
                    resampled_species_matrix.append(resampled_species_vector)
                    resampled_enzyme_matrix.append(resampled_enzyme_vector)

                surf_fig = plt.figure(figsize=(11, 7))
                surf_ax = surf_fig.gca(projection='3d')
                X = np.array(resampled_species_matrix)
                Y = np.array(resampled_enzyme_matrix)
                Z = interpolate.griddata((np.ravel(species_matrix), np.ravel(enzyme_matrix)), np.ravel(fraction_matrix), (X, Y), method='linear')

                X = np.ravel(X)
                Y = np.ravel(Y)
                Z = np.ravel(Z)

                x = np.full_like(X, 0.8)
                y = np.full_like(Y, 0.4)
                z = np.full_like(Z, 0)

                fracs = np.ravel(X.astype(float))/np.ravel(X.max())
                norm = colors.Normalize(fracs.min(), fracs.max())
                color_values = cm.coolwarm(norm(fracs.tolist()))

                surf_ax.bar3d(Y, X, z, y, x, Z, color=color_values, edgecolor='w', linewidth=0.1, shade=False)
                surf_ax.set_ylabel('RNA polyA length', labelpad=10, fontweight='bold')
                surf_ax.set_xlabel(fr'[{sample_name}] $\mu$M', labelpad=10, fontweight='bold')
                surf_ax.set_zlabel('Fraction', labelpad=10, fontweight='bold')
                surf_ax.set_yticks(np.linspace(2, kinetic_model.n, 9))
                surf_ax.set_xticks(np.linspace(Y.min(), Y.max(), 10))
                surf_ax.set_ylim([X.max()+0.05 * X.max(), X.min()-0.05 * X.max()])
                surf_ax.set_xlim([Y.min()-0.05 * Y.max(), Y.max()+0.05 * Y.max()])
                surf_ax.set_zlim([-0.01, 1.01])
                surf_ax.set_title(f"Time: {time} s")

                surf_ax.xaxis.pane.fill = False
                surf_ax.yaxis.pane.fill = False
                surf_ax.zaxis.pane.fill = False
                surf_ax.xaxis.pane.set_edgecolor('w')
                surf_ax.yaxis.pane.set_edgecolor('w')
                surf_ax.zaxis.pane.set_edgecolor('w')

                surf_ax.set_box_aspect(aspect=(1.8,1.8,1))
                surf_ax.view_init(azim=-60, elev=30)
                surf_fig.tight_layout()
                surf_figs.append(surf_fig)
                plt.close(surf_fig)

            for fig in surf_figs:
                pdf.savefig(fig)
        plt.close()

    def run_plots(self):

        if self.best_fit_flag == True:
            self.plot_best_fit(self.experiments, self.kinetic_models, self.hybridization_models, self.enzyme_colors, self.sample_name, self.pdf, self.best_fit_flag, self.plot_mean_flag, self.xlimits, self.marker_shape)

            if self.residual_flag == True:
                self.plot_residuals(self.experiments, self.kinetic_models, self.residuals, self.enzyme_colors, self.pdf, self.plot_mean_flag, self.xlimits, self.marker_shape)
                self.plot_combined_residuals(self.experiments, self.kinetic_models, self.residuals, self.enzyme_colors, self.pdf, self.plot_mean_flag, self.xlimits, self.marker_shape)

            if self.RNA_populations_flag == True:
                self.plot_rna_populations(self.experiments, self.kinetic_models, self.sample_name, self.pdf, self.output_dir, self.export_ss_times, self.xlimits)
            
            if self.RNA_populations_3D_flag == True:
                self.plot_rna_populations_3D(self.experiments, self.kinetic_models, self.sample_name, self.pdf, self.enzyme_colors, self.xlimits)

            if self.enzyme_populations_flag == True:
                self.plot_enzyme_populations(self.experiments, self.kinetic_models, self.sample_name, self.pdf, self.xlimits)
            
            if self.annealed_fraction_flag == True:
                self.plot_annealed_fraction(self.experiments, self.hybridization_models, self.enzyme_colors, self.sample_name, self.pdf)
                    
            if self.bar_2d_flag == True:
                self.plot_2d_population_bars(self.experiments, self.kinetic_models, self.enzyme_bar_colors, self.t_pop_colors, self.pdf, self.timesample)
                    
            if self.bar_3d_flag == True:
                self.plot_3d_population_bars(self.experiments, self.kinetic_models, self.hybridization_models, self.pdf, self.timesample, self.sample_name)                
            self.pdf.close()

    def get_colors(self, points=100, colormap=cm.coolwarm, map_name='plot_colors', trim = True):
        if trim == True:
            color_values = colormap(np.linspace(0, 1, points+2))
            color_values = color_values[1:-1]
        else:
            color_values = colormap(np.linspace(0, 1, points))
            color_values = color_values
        self.addattr(map_name, color_values)

def make_pdf(pdf_name):
    pdf = PdfPages(pdf_name)
    return pdf


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    if any(exponent == x for x in [0, 1, -1]):
        return r"${0:.{2}f}$".format(num, exponent, precision)
    else:
        return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)




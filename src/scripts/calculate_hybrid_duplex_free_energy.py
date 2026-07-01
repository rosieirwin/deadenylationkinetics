#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Calculate the free energy (ΔG) of RNA/DNA hybrid duplexes with varying lengths of polyA tails.

Ref. Banerjee, D. et al. Improved nearest-neighbor parameters for the stability of RNA/DNA hybrids under
a physiological condition. Nucleic Acids Res. 48(21):12042-54. 2020.
"""

#### User inputs ####

L = 1e-7 # ligand concentration in M
n = 18 # number of As in polyA tail
RNA_tag_sequence = 'CCUUUCC' # if there are any bases preceeding the polyA tail, include them here
temp_C = 25 # C
salt = 'low' # 'low' (~100 mM) or 'high' (~1 M) salt conditions
output_dir = f'duplex_tag-{RNA_tag_sequence}_n-{n}_conc-{L}M_{temp_C}C_{salt}_salt'

######################


def main(L, n, RNA_tag_sequence, temp_C, salt, output_dir):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    temperature = temp_C + 273.15  # K
    polyA_sequence = [i*'A' for i in range(0,n+1)]
    numA = [i for i in range(0,n+1)]
    RNA_sequences = [RNA_tag_sequence + i for i in polyA_sequence]
    
    nn_pairs = make_nn_pairs(RNA_sequences)
    nn_thermo = nn_thermodynamics()
    
    dG_dict = calculate_dG(nn_pairs, numA, temperature, nn_thermo, salt)
    binding_dict = fraction_bound(dG_dict, L, temperature, salt, temp_C, output_dir)

    write_dG_csv(dG_dict, temperature, salt, temp_C, output_dir)
    plot_df = plot_dict(dG_dict, binding_dict, temperature)
    plot_dG(plot_df, salt, temp_C, output_dir)
    plot_Kd_frac(plot_df, salt, temp_C, output_dir)


def make_nn_pairs(sequences):
    """
    Creates a dictionary of nearest-neighbor pairs for each RNA sequence.
    """
    nn_pairs = {sequence:[sequence[i:i+2] for i in range(len(sequence[:-1]))] for sequence in sequences}

    return nn_pairs


def nn_thermodynamics():
    """
    Defines the nearest-neighbor thermodynamic parameters for RNA/DNA hybrid duplexes.
    """
    nn_pairs = ['rAA/dTT','rAC/dGT','rAG/dGT','rAU/dAT','rCA/dTG','rCC/dGG','rCG/dCG',
    'rCU/dAG','rGA/dTC','rGC/dGC','rGG/dCC','rGU/dAC','rUA/dTA','rUC/dGA','rUG/dCA','rUU/dAA']
    nn_dH = [-7.8,-10.1,-9.4,-5.8,-9.8,-9.5,-9.0,-6.1,-8.6,-10.6,-13.3,-9.3,-6.6,-6.5,-8.9,-7.4] # kcal/mol
    nn_dS = [-22.9,-27.3,-26.2,-17.5,-27.4,-24.8,-24.3,-17.9,-22.7,-27.7,-35.7,-25.5,-19.7,-16.3,-23.3,-24.3] # cal /mol /K
    nn_thermo = {nn_pairs[i]:{'dH':nn_dH[i],'dS':nn_dS[i]} for i in range(len(nn_pairs))}
    
    return nn_thermo


def calculate_dG(nn_pairs, numA, temperature, nn_thermo, salt):
    """
    Calculates the free energy (ΔG) of RNA/DNA hybrid duplexes.
    """
    dG_dict = {sequence:{'dG_kcal':None,'dG_kJ':None,'numA':numA[i], 'numNT':len(sequence)} for i,sequence in enumerate(nn_pairs.keys())}
    for sequence in nn_pairs.keys():

        dH_sum = 0
        dS_sum = 0

        for nn_pair in nn_pairs[sequence]:

            for k in nn_thermo.keys():

                if 'r' + nn_pair + '/' in k:
            
                    dH_sum += nn_thermo[k]['dH']
                    dS_sum += nn_thermo[k]['dS']

        dG = dH_sum - temperature*(dS_sum/1000) # /1000 to convert cal /mol /K to kcal /mol /K to match dH
        if salt == 'high':
            dG = (dG + 1.667)/0.63
        dG_dict[sequence]['dG_kcal'] = dG # kcal / mol
        dG_dict[sequence]['dG_kJ'] = dG*4.184 # kJ / mol
    

    return dG_dict


def fraction_bound(dG_dict, L, temperature, salt, temp_C, output_dir):
    """
    Determines the fraction of RNA/DNA hybrid duplexes that are bound at a given ligand concentration.
    """
    R = 1.987e-3 # kcal / mol / K

    K_dict = {k:np.exp(dG_dict[k]['dG_kcal']/(R*temperature)) for k in dG_dict.keys()}

    maxy = 1
    fraction_bound_dict = {k:(maxy*L)/(L+K_dict[k]) for k in K_dict.keys()}
    binding_dict = {k:{'Kd':K_dict[k],'fraction bound':fraction_bound_dict[k]} for k in K_dict.keys()}

    csv_dict = {'polyA length':[dG_dict[k]['numA'] for k in dG_dict.keys()],'Kd M':[K_dict[k] for k in K_dict.keys()]}
    csv_df = pd.DataFrame(csv_dict)

    csv_df.to_csv(f'{output_dir}/RNA_DNA_hybrid_duplex_{temp_C}C_{salt}_salt_Kd.csv',index=False)

    return binding_dict


def write_dG_csv(dG_dict, temperature, salt, temp_C, output_dir):
    """
    Writes the free energy (ΔG) data of RNA/DNA hybrid duplexes to a CSV file.
    """
    csv_dict = {'RNA/DNA duplex sequence':[k for k in dG_dict.keys()],'dG kcal/mol':[dG_dict[k]['dG_kcal'] for k in dG_dict.keys()],
                'dG kJ/mol':[dG_dict[k]['dG_kJ'] for k in dG_dict.keys()],'Temperature K':[temperature for k in dG_dict.keys()],
                'sequence length':[dG_dict[k]['numNT'] for k in dG_dict.keys()],'polyA length':[dG_dict[k]['numA'] for k in dG_dict.keys()]}
    csv_df = pd.DataFrame(csv_dict)
    
    filename = f'{output_dir}/RNA_DNA_hybrid_duplex_{temp_C}C_{salt}_salt_Go.csv'
    csv_df.to_csv(filename,index=False)


def plot_dict(dG_dict, binding_dict, temperature):
    plot_dict = {'RNA/DNA duplex sequence':[k for k in dG_dict.keys()],'dG kcal/mol':[dG_dict[k]['dG_kcal'] for k in dG_dict.keys()],
                 'dG kJ/mol':[dG_dict[k]['dG_kJ'] for k in dG_dict.keys()],'Temperature K':[temperature for k in dG_dict.keys()],
                 'sequence length':[dG_dict[k]['numNT'] for k in dG_dict.keys()],'polyA length':[dG_dict[k]['numA'] for k in dG_dict.keys()],
                 'Kd':[binding_dict[k]['Kd'] for k in binding_dict.keys()], 'fraction bound':[binding_dict[k]['fraction bound'] for k in binding_dict.keys()]}
    plot_df = pd.DataFrame(plot_dict)

    return plot_df


def plot_dG(plot_df, salt, temp_C, output_dir):
    """
    Plots the free energy (ΔG) of RNA/DNA hybrid duplexes.
    """
    FL = plot_df['sequence length'].iloc[-1]
    n = plot_df['polyA length'].iloc[-1]
    units = ['kcal','kJ']
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    for ia,a in enumerate(ax):
        a.plot(plot_df['sequence length'],plot_df[f'dG {units[ia]}/mol'],color='k',marker='o',linestyle='none')
        
        # Perform linear fit
        fit_coeffs = np.polyfit(plot_df['sequence length'][plot_df['polyA length'] > 0], plot_df[f'dG {units[ia]}/mol'][plot_df['polyA length'] > 0], 1)
        fit_line = np.poly1d(fit_coeffs)
        # Plot the linear fit
        a.plot(plot_df['sequence length'], fit_line(plot_df['sequence length']), color='r', linestyle='--', label='Linear Fit')
        a.text(0.95, 0.95, r'Fit: $\Delta G_{i} = \alpha i + \Delta G_{0}$', transform=a.transAxes, ha='right', va='top', color='r')
        a.text(0.95, 0.9, fr'$\Delta G_{{i}} = {fit_coeffs[0]:.4f} i$ + {fit_coeffs[1]:.4f}', transform=a.transAxes, ha='right', va='top', color='r')
        a.set_xlabel('Number of NTs (# As)', fontsize=12)
        a.tick_params(axis='both', direction='in')
        a.set_xticks(range(0,FL+1,5))
        a.set_xlim(-1,FL+1)
        tick_labels = [f'{s} (0)' for s in range(0,FL-n)]
        tick_labels = tick_labels + [f'{s} ({a})' for s, a in zip(plot_df['sequence length'],plot_df['polyA length'])]
        a.set_xticklabels([tl for i,tl in enumerate(tick_labels) if i % 5 == 0],rotation=45)
        a.set_ylabel(fr'Duplex $\Delta G$ ({units[ia]}/mol)', fontsize=12)
    fig.suptitle(f"RNA/DNA duplex free energy vs. polyA length @ {plot_df['Temperature K'][0]} K")
    fig.tight_layout()

    filename = f'{output_dir}/RNA_DNA_hybrid_duplex_{temp_C}C_{salt}_salt_Go.pdf'
    plt.savefig(filename)


def plot_Kd_frac(plot_df, salt, temp_C, output_dir):
    """
    Plots the dissociation constant (Kd) and fraction bound of RNA/DNA hybrid duplexes.
    """
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.55, 0.3, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    ax1.plot(plot_df['polyA length'],plot_df['fraction bound'],color='k',marker='o',linestyle='none',label='RNA-DNA duplex')
    ax1.plot(plot_df['polyA length'],1-plot_df['fraction bound'],color='grey',marker='^',linestyle='none',label='Free RNA')
    ax1.set_xticks(range(0,21,4))
    ax1.set_xlabel('RNA tag with # As')
    ax1.set_ylabel('Fraction')
    ax1.tick_params(axis='both', direction='in')
    ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 0.4, 0.5, 0.5),frameon=False)


    ax2.plot(plot_df['polyA length'],plot_df['Kd'],color='k',marker='.',linestyle='none')
    ax2.set_xticks(range(0,21,4))
    ax2.set_yscale('log')
    ax2.set_xlabel('RNA tag with # As')
    ax2.set_ylabel(r'Duplex $K_{D}$ (M)')
    ax2.tick_params(axis='both', direction='in')

    filename = f'{output_dir}/RNA_DNA_hybrid_duplex_{temp_C}C_{salt}_salt_Kd.pdf'
    plt.savefig(filename)


main(L, n, RNA_tag_sequence, temp_C, salt, output_dir)

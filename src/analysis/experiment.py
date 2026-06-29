#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors: Robert W. Harkness, Rose M. Irwin
"""

import pandas as pd

class FretExperiment():

    def __init__(self, data, hybridization_params, config_params):
        self.data = data
        self.data_groups = data.groupby('Enzyme')
        self.time = []
        self.fret = []
        self.rna_all = []
        self.dna_all = []
        self.unique_time = []
        self.mean_fret = []
        self.fret_std = []
        self.mean_resid=[]
        self.resid_std=[]
        self.enzyme = data.Enzyme.unique()
        self.rna = data.RNA.unique().astype(float) 
        self.QT = data.DNA.unique().astype(float)[0] # For now, can only have one DNA concentration
        self.n = hybridization_params['n']
        self.dGo = hybridization_params['dGo']
        self.alpha = hybridization_params['alpha']
        self.temperature = hybridization_params['Temperature']

        if config_params['Modeling parameters']['SS time range']['Limit'] == True:
            ss_times = pd.read_csv(config_params['Modeling parameters']['SS time range']['File'])
        
        if any(config_params['Modeling parameters']['Kinetic model'] == x for x in ['Distributive-dense','2phase SS distributive']):
            self.partitioning_data = pd.read_csv(config_params['Partitioning']['Data file'])
            self.use_dvf = config_params["Partitioning"]["Use DVF"]

        for idx, (ind, group) in enumerate(self.data_groups): # Convert data frame into list-of-lists of time, fret, RNA, and DNA
            if config_params['Modeling parameters']['SS time range']['Limit'] == False:
                self.time.append(group.Time.values)
                self.fret.append(group.FRET.values)
                self.rna_all.append(group.RNA.values)
                self.dna_all.append(group.DNA.values)
            else:
                time_mask = group.Time.values <= ss_times['SteadyStateTime'][idx]
                count = sum(time_mask)
                if count < 3:
                    print(f'Warning: Not enough data points for steady state analysis of {ind}. Skipping data set.')
                    self.enzyme = self.enzyme[self.enzyme != ind]
                else:
                    self.time.append(group.Time.values[time_mask])
                    self.fret.append(group.FRET.values[time_mask])
                    self.rna_all.append(group.RNA.values[time_mask])
                    self.dna_all.append(group.DNA.values[time_mask])


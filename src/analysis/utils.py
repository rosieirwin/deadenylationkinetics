#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors: Robert W. Harkness, Rose M. Irwin
"""

import pandas as pd
import yaml
import os
import numpy as np


def load_data(configuration_file):
    config_params = yaml.safe_load(open(configuration_file,'r'))
    replicate_df = pd.read_csv(config_params['Data file to fit'])
    return config_params, replicate_df


def setup_parameters(config_params, initial_guess_params):

    hybridization_params = {k:config_params['Experimental parameters'][k]['Value'] for k in ['n', 'Temperature']}
    hybridization_params['dGo'] = config_params['Modeling parameters']['Fit parameters']['dGo']['Value']
    hybridization_params['alpha'] = config_params['Modeling parameters']['Fit parameters']['alpha']['Value']

    if "2phase" in config_params["Modeling parameters"]["Kinetic model"]:
        varied_params, opt_params_dict, opt_param_units_dict = [], {}, {}
        all_param_units = []
        for phase in ["Dilute", "Dense"]:

            opt_params_dict[phase] = {}
            opt_param_units_dict[phase] = []
            for k in config_params['Modeling parameters']['Fit parameters'][f"{phase} parameters"].keys():
                all_param_units.append(config_params['Modeling parameters']['Fit parameters'][f"{phase} parameters"][k]['Units'])
                initial_guess_params.add(f"{k}_{phase}", value = config_params['Modeling parameters']['Fit parameters'][f"{phase} parameters"][k]['Value'], 
                                                     vary = config_params['Modeling parameters']['Fit parameters'][f"{phase} parameters"][k]['Vary'], 
                                                     min = float(config_params['Modeling parameters']['Fit parameters'][f"{phase} parameters"][k]['Minimum']), 
                                                     max = float(config_params['Modeling parameters']['Fit parameters'][f"{phase} parameters"][k]['Maximum']))
                if config_params['Modeling parameters']['Fit parameters'][f"{phase} parameters"][k]['Vary'] == True:
                    varied_params.append(f"{k}_{phase}")
                    opt_params_dict[phase][k] = []
                    opt_param_units_dict[phase].append(config_params['Modeling parameters']['Fit parameters'][f"{phase} parameters"][k]['Units'])
        for phase in ["Dilute", "Dense"]:
            if len(opt_params_dict[phase]) == 0:
                del opt_params_dict[phase]
                del opt_param_units_dict[phase]
        return  hybridization_params, initial_guess_params, all_param_units, varied_params, opt_params_dict, opt_param_units_dict
    else:
        for k in config_params['Modeling parameters']['Fit parameters'].keys():
            initial_guess_params.add(k, value = config_params['Modeling parameters']['Fit parameters'][k]['Value'], 
                                     vary = config_params['Modeling parameters']['Fit parameters'][k]['Vary'], 
                                     min = float(config_params['Modeling parameters']['Fit parameters'][k]['Minimum']), 
                                     max = float(config_params['Modeling parameters']['Fit parameters'][k]['Maximum']))
        varied_params = [k for k in config_params['Modeling parameters']['Fit parameters'].keys() if config_params['Modeling parameters']['Fit parameters'][k]['Vary'] == True]
        opt_params = {k:[] for k in config_params['Modeling parameters']['Fit parameters'].keys() if config_params['Modeling parameters']['Fit parameters'][k]['Vary'] == True}
        opt_param_units = [config_params['Modeling parameters']['Fit parameters'][k]['Units'] for k in config_params['Modeling parameters']['Fit parameters'].keys() if config_params['Modeling parameters']['Fit parameters'][k]['Vary'] == True]
        all_param_units = [config_params['Modeling parameters']['Fit parameters'][k]['Units'] for k in config_params['Modeling parameters']['Fit parameters'].keys()]

        return  hybridization_params, initial_guess_params, all_param_units, varied_params, opt_params, opt_param_units


def create_experiment_dataframe(time, avgFRET, stdFRET, E0, S0):
    data_dict = {'Time':[],'FRET':[],'Error':[], 'Enzyme':[], 'RNA':[]} # Input data is list of lists, needs to be unraveled into one list for making data frame
    data_dict['Time'] = [t for time_vector in time for t in time_vector]
    data_dict['FRET'] = [fret for fret_vector in avgFRET for fret in fret_vector]
    data_dict['Error'] = [fret for fret_vector in stdFRET for fret in fret_vector]
    data_dict['Enzyme'] = [E0[i] for i,time_vector in enumerate(time) for t in time_vector] # Match each time point with corresponding enzyme and RNA concentration
    data_dict['RNA'] = [S0[i] for i,time_vector in enumerate(time)  for t in time_vector]
    return pd.DataFrame(data_dict)


def make_dictionary(key_list, value_list):
    made_dictionary = {k:v for k,v in zip(key_list,value_list)}
    return made_dictionary

def write_optimal_parameter_csv(param_values, param_units, config_params, varied_params, rmsd):
    file = config_params['Optimal fit parameter file']
    output_dir = config_params['Output location']
    model = config_params['Modeling parameters']['Kinetic model']
    if model == "2phase":
        
        params_dict = {'Parameter':[k for k in param_values.keys() if 'Error' not in k], 'Value':[f"{param_values[k]:.6e}" for k in param_values.keys() if 'Error' not in k],
                       'Units':[i for i in param_units], 'Varied':[True if k in varied_params else False for k in param_values.keys() if 'Error' not in k]}
    else:
        params_dict = {'Parameter':[k for k in param_values.keys() if 'Error' not in k], 'Value':[f"{param_values[k]:.6e}" for k in param_values.keys() if 'Error' not in k],
                       'Units':[i for i in param_units], 'Varied':[True if k in varied_params else False for k in param_values.keys() if 'Error' not in k]}

    params_dict['Parameter'].append('RMSD')
    params_dict['Value'].append(float(rmsd))
    params_dict['Units'].append('n/a')
    params_dict['Varied'].append('n/a')

    if "2phase" in model:
        for phase in ["Dilute","Dense"]:
            params_dict['Parameter'].append(f'kobs_{phase}')
            if any(config_params['Modeling parameters']['Kinetic model']== x for x in ['Steady state distributive', '2phase steady state distributive', '2phase SS distributive']):
                params_dict['Value'].append(float(param_values[f'kcat_{phase}']/param_values[f'KM_{phase}']))
            else:
                params_dict['Value'].append(calc_kobs(param_values[f'k1_{phase}'],param_values[f'km1_{phase}'],param_values[f'k2_{phase}'],param_values[f'km2_{phase}'],param_values[f'kcat_{phase}']))
            params_dict['Units'].append('M^-1s^-1')
            params_dict['Varied'].append('n/a')
    else:
        params_dict['Parameter'].append('kobs')
        if any(model == x for x in ['Steady state distributive', '2phase steady state distributive', '2phase SS distributive']):
            params_dict['Value'].append(float(param_values['kcat']/param_values['KM']))
        else:
            params_dict['Value'].append(calc_kobs(param_values['k1'],param_values['km1'],param_values['k2'],param_values['km2'],param_values['kcat']))
        params_dict['Units'].append('M^-1s^-1')
        params_dict['Varied'].append('n/a')
    params_df = pd.DataFrame(params_dict)
    params_df.to_csv(f"{output_dir}/{file}")

def write_parameter_RSS_csv(initial_guess_params, output_dir, sum_of_squared_residuals, resids, config_params):
        
        param_names = [str(param) for param in initial_guess_params.keys()]
        param_values = [initial_guess_params[param].value for param in initial_guess_params.keys()]
        if any(config_params["Modeling parameters"]["Kinetic model"] == x for x in ['Steady state distributive', '2phase steady state distributive', '2phase SS distributive']):
            param_values.append(float(param_values[0]/param_values[1]))
        else:
            param_values.append(calc_kobs(param_values[0],param_values[1],param_values[2],param_values[3],param_values[4]))
        param_values.append(float(sum_of_squared_residuals(resids[0])))
        param_values.append(len(np.concatenate(resids[0], axis=None)))
        param_names_string = ','.join(param_names)+',kobs,RSS,len'
        param_values_string = ','.join([str(v) for v in param_values])
        
        # Check if RSS_simulated_data.csv exists
        rss_file = os.path.join(output_dir, 'RSS_simulated_data.csv')
        if os.path.exists(rss_file):
            try:
                with open(rss_file, 'a') as f:
                    f.write(f'{param_values_string}\n')
            except Exception as e:
                print(f'Error appending RSS for simulated data: {e}')
        else:
            # Save RSS for simulated data in .csv
            try:
                with open(rss_file, 'w') as f:
                    f.write(f'{param_names_string}\n')
                    f.write(f'{param_values_string}\n')
            except Exception as e:
                print(f'Error saving RSS for simulated data: {e}')

def calc_kobs(k1,km1,k2,km2,kcat):
    return float(kcat / (((km2 + kcat) / k2) * (1 + (km1 / k1))))
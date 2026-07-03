#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors: Robert W. Harkness, Rose M. Irwin
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from lmfit import minimize, Parameters, report_fit
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from minimization import sum_of_squared_residuals
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm
import os

class ErrorAnalysis():

    def __init__(self, opt_params, varied_params = None, cores=None, monte_carlo_iterations=None, rmsd=None, range_factor=None, points=None, export_sim_data = False, opt_param_units=None):

        self.opt_params = opt_params
        for ki, k in enumerate(opt_param_units):
            if any(k in s for s in ['Dilute', 'Dense']):
                for ui, u in enumerate(opt_param_units[k]):
                    if opt_param_units[k] == None:
                        opt_param_units[k] = ['']*len(opt_params)
                    else:
                        opt_param_units[k][ui] = u.replace('-1','{-1}')    
            else:
                if opt_param_units == None:
                    opt_param_units = ['']*len(opt_params)
                else:
                    opt_param_units[ki] = k.replace('-1','{-1}')
        self.varied_params = varied_params
        self.param_units = opt_param_units
        self.monte_carlo_iterations = monte_carlo_iterations # For Monte carlo
        self.rmsd = rmsd
        self.range_factor = range_factor # For correlation surfaces
        self.points = points
        if cores == 0:
            self.cores = cpu_count() - 1
        else:
            if cores <= cpu_count():
                self.cores = cores
            else:
                self.cores = cpu_count() - 1
        self.export_sim_data = export_sim_data


    ####### Parameter correlations #######
        
    @staticmethod
    def parameter_range(opt_param, scaling_factor=10.0, num_points=10):
        # Generate parameter vectors for correlation surfaces
        if opt_param < 0:
            opt_param_range = np.linspace(opt_param/scaling_factor, opt_param*scaling_factor, num_points)
        else:
            opt_param_range = np.logspace(np.log10(opt_param/scaling_factor), np.log10(opt_param*scaling_factor), num_points) # +/- scaling factor orders of magnitude from optimal value
        
        return opt_param_range
    
    def correlation_pairs(self):
        self.correlation_pairs = {} # Big dictionary of all parameter pair combinations and their associated Parameters objects for passing to fitting routine
        params_to_correlate = [k for k in self.varied_params]
        from copy import deepcopy
        opt_params_copy = deepcopy(self.opt_params)

        for i in range(len(params_to_correlate) - 1): # Need to correlate ith parameter with only the parameters ahead of it, don't need to do last parameter because it gets done along the way
            param_1_range = self.parameter_range(self.opt_params[params_to_correlate[i]].value, self.range_factor, self.points)
            for j in range(i + 1, len(params_to_correlate)):
                param_2_range = self.parameter_range(self.opt_params[params_to_correlate[j]].value, self.range_factor, self.points)
                self.correlation_pairs[f"{params_to_correlate[i]},{params_to_correlate[j]}"] = {f"{params_to_correlate[i]}":[], f"{params_to_correlate[j]}":[], "Parameter sets":[], "RSS":[], 'Fit results':[], 'Result order':[]}
                for k, param_1 in enumerate(param_1_range): # Iterate over values for each parameter pairing, set the pairs in question to constants, allow params not in correlation pair to be varied
                    for l, param_2 in enumerate(param_2_range):
                        opt_params_copy[params_to_correlate[i]].value = param_1
                        opt_params_copy[params_to_correlate[i]].vary = False
                        opt_params_copy[params_to_correlate[j]].value = param_2
                        opt_params_copy[params_to_correlate[j]].vary = False
                        self.correlation_pairs[f"{params_to_correlate[i]},{params_to_correlate[j]}"]["Parameter sets"].append(opt_params_copy) # Parallel fit results are not in the same order as this
                        opt_params_copy = deepcopy(self.opt_params)

    def parameter_correlation_fits(self, experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper):
        maxParallelProcesses = self.cores
        print(f'\n### Running parameter correlation fits using {maxParallelProcesses} CPU cores. ###')
        for param_pairs in self.correlation_pairs.keys():
            parameter_sets = self.correlation_pairs[param_pairs]['Parameter sets']
            print(f'Running parameter pair {param_pairs}.')
            with ProcessPoolExecutor(max_workers = maxParallelProcesses) as parallelExecution:
                future_results = {}
                with tqdm(total=len(parameter_sets), desc=f"{param_pairs} progress") as pbar:
                    for x in list(np.arange(len(parameter_sets))):
                        future_result = parallelExecution.submit(self.parallel_fit_task, parameter_sets[x], experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper)
                        future_results[future_result] = x
                    for future in as_completed(future_results):
                        ax = future_results[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            print(f'{ax} generated an exception in parameter correlation fits: {exc}')
                        else:
                            pbar.update(1)
                            self.correlation_pairs[param_pairs]['Result order'].append(ax)
                            self.correlation_pairs[param_pairs]['Fit results'].append(result.params)
                            self.correlation_pairs[param_pairs]['RSS'].append(result.chisqr)
                            self.correlation_pairs[param_pairs][param_pairs.split(',')[0]].append(result.params[param_pairs.split(',')[0]].value)
                            self.correlation_pairs[param_pairs][param_pairs.split(',')[1]].append(result.params[param_pairs.split(',')[1]].value)
        

    def parameter_correlation_surfaces(self, config_params):
        sample_name = config_params['Sample name']
        output_dir = config_params['Output location']

        pdf = make_pdf(f"./{output_dir}/{sample_name}_parameter-correlation-surfaces_range-{self.range_factor}_points-{self.points}.pdf")

        for param_pairs in self.correlation_pairs.keys():
            param_pair_values = self.correlation_pairs[param_pairs]
            sorted_values = sorted(zip(*[param_pair_values[k] for k in param_pair_values]), key=lambda x: x[-1])
            sorted_param_pair_values = {k: [v[i] for v in sorted_values] for i, k in enumerate(param_pair_values)}

            x = sorted_param_pair_values[param_pairs.split(',')[0]]
            y = sorted_param_pair_values[param_pairs.split(',')[1]]
            z = sorted_param_pair_values['RSS']
            try:
                xgrid, ygrid, zgrid = self.make_grid_data(x, y, z, resolution=self.points)

                fig, ax = plt.subplots(1, 1)
                a = ax.contourf(xgrid, ygrid, zgrid, levels=100, cmap='turbo')
                ax.plot(self.opt_params[param_pairs.split(',')[0]].value, self.opt_params[param_pairs.split(',')[1]].value, 'X', markersize=10, mew=1, mec='k', mfc='w', label="Optimal Parameters")
                cbar = fig.colorbar(a, format='%.2e')
                cbar.ax.set_title('RSS', pad=10)
                x_label = param_pairs.split(',')[0]
                y_label = param_pairs.split(',')[1]
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend()
                ax.set_title(f"Parameter correlation surface for {param_pairs}")
                pdf.savefig(fig)
                plt.close()
            except Exception as exc:
                print(f'Error occurred while making grid data for {param_pairs}: {exc}')
        pdf.close()

    @staticmethod
    def make_grid_data(x, y, z, resolution=100, contour_method='linear'):
        x_resample = np.linspace(min(x), max(x), resolution)
        y_resample = np.linspace(min(y), max(y), resolution)
        x_grid, y_grid = np.meshgrid(x_resample, y_resample)
        z_grid = griddata((x, y), z, (x_grid, y_grid), contour_method)
        return x_grid, y_grid, z_grid
    
    def save_parameter_correlation_results(self, config_params):
        sample_name = config_params['Sample name']
        model = config_params['Modeling parameters']['Kinetic model']
        output_dir = config_params['Output location']

        result_dict = {key: [] for key in self.correlation_pairs[list(self.correlation_pairs.keys())[0]]['Fit results'][0].keys()}
        result_dict['RSS'], result_dict['Result order'], result_dict['Parameter pair'] = [], [], []
        for ppi, param_pairs in enumerate(self.correlation_pairs.keys()):
            param1 = param_pairs.split(',')[0]
            param2 = param_pairs.split(',')[1]
            for key in self.correlation_pairs[param_pairs]['Fit results'][0].keys():
                if key == param1 or key == param2:
                    result_dict[key].append(self.correlation_pairs[param_pairs][key])
                else:
                    result_dict[key].append([x[key].value for x in self.correlation_pairs[param_pairs]['Fit results']])
            result_dict['RSS'].append(self.correlation_pairs[param_pairs]['RSS'])
            result_dict['Result order'].append(self.correlation_pairs[param_pairs]['Result order'])
            result_dict['Parameter pair'].append([param_pairs] * len(self.correlation_pairs[param_pairs]['RSS']))
        
        merged_result_dict = {}
        for key in result_dict.keys():
            temp_list = []
            temp_list.append([item for sublist in result_dict[key] for item in sublist])
            merged_result_dict[key] = temp_list[0]
        merged_result_df = pd.DataFrame(merged_result_dict)
        merged_result_df.to_csv(f"{output_dir}/{sample_name}_parameter-correlation-results_range-{self.range_factor}_points-{self.points}.csv")


    ###### Monte Carlo #######

    def monte_carlo_parameter_dictionary(self):
        init_params = Parameters()
        self.monte_carlo_parameters = {k:[] for k in self.opt_params.keys() if self.opt_params[k].vary == True}
        self.monte_carlo_rss = []
        self.monte_carlo_data = []
        self.monte_carlo_errors = {f"{k} error":None for k in self.opt_params.keys() if self.opt_params[k].vary == True}
        self.hist = {k: {'pretty name':[],'hist_data':[],'counts': [], 'bin_edges': [], 'bin_centers': [],'fit_params': init_params, 'fit_result':[], 'x_sim':[], 'y_sim': []} for k in self.opt_params.keys()}

    def monte_carlo_fits(self, experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper):
        maxParallelProcesses = self.cores
        print(f'\n### Running Monte Carlo fits using {maxParallelProcesses} CPU cores. ###')
        with ProcessPoolExecutor(max_workers = maxParallelProcesses) as parallelExecution:
            future_results = {}
            with tqdm(total=self.monte_carlo_iterations, desc="Monte Carlo progress") as pbar:
                for x in list(np.arange(1, self.monte_carlo_iterations + 1)):
                    future_result = parallelExecution.submit(self.monte_carlo_parallel_fit_task, self.opt_params, experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper, self.rmsd, x, self.export_sim_data)
                    future_results[future_result] = x
                for future in as_completed(future_results):
                    ax = future_results[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print('\n%r generated an exception in Monte Carlo fits: %s' % (ax, exc))
                    else:
                        {self.monte_carlo_parameters[k].append(result.params[k].value) for k in self.monte_carlo_parameters.keys()}
                        self.monte_carlo_rss.append(sum_of_squared_residuals(result.residual))
                        pbar.update(1)
        for k in self.monte_carlo_parameters.keys():
            self.monte_carlo_errors[f"{k} error"] = np.std(self.monte_carlo_parameters[k])
        print('\n### Monte Carlo parameter error estimates ###')
        for k1, k2 in zip(self.monte_carlo_parameters.keys(), self.monte_carlo_errors.keys()):
            print(f"{k1} = {self.opt_params[k1].value} +/- {self.monte_carlo_errors[k2]}")

    @staticmethod
    def parallel_fit_task(initial_guess_params, experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper, min_method='leastsq', print_current_params=False):
        minimizer_result = minimize(objective_wrapper, initial_guess_params, method = min_method, args=(experiment, kinetic_model, hybridization_model, simulate_full_model, print_current_params))
        return minimizer_result
    
    @staticmethod
    def monte_carlo_parallel_fit_task(initial_guess_params, perfect_experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper, rmsd, xi, export_sim_data=False, min_method='leastsq', print_current_params=False):
        perturbed_experiment = deepcopy(perfect_experiment)
        for i,v in enumerate(perturbed_experiment.fret):
            perturbed_experiment.fret[i] = v + np.random.RandomState().normal(scale=rmsd,size=np.size(v, 0))
            # if indicated, simulated data will be exported from the models.py file
        
        perturbed_minimizer_result = minimize(objective_wrapper, initial_guess_params, method = min_method, 
        args=(perturbed_experiment, kinetic_model, hybridization_model, simulate_full_model, print_current_params))
        return perturbed_minimizer_result

    def gaussian(self, x, amp, cen, wid):
        # f(x)= C * e ^ {-(x-x_mean)^2/(2sigma^2)}
            # amp = height of the curve's peak
            # cen = the position of the center of the peak
            # wid = the standard deviation, controls the width of the "bell"
            # f(x) = function of x
            # e	= Euler's number
            # x	= integer
        y = (amp/(wid*np.sqrt(2*np.pi)))*np.exp(-(1/2)*np.square((x - cen)/wid))
        return y

    def gauss_objective(self, fit_params, x, y): # Minimization function for global fit
        amp = fit_params['amp'].value
        cen = fit_params['cen'].value
        wid = fit_params['wid'].value
        y_sim = self.gaussian(x, amp, cen, wid)
        resid = y - y_sim
        return resid

    def monte_carlo_distributions(self):
        keys = [k for k in self.monte_carlo_parameters.keys()]
        mc_iter = len(self.monte_carlo_parameters)
        for k in keys:
            try:
                self.hist[k]['pretty name'] = f"$k_{{{k[1:]}}}$"
                self.hist[k]['hist_data'] = np.log10(self.monte_carlo_parameters[k])
                counts, edges = np.histogram(self.hist[k]['hist_data'], bins=100) # get histogram of log(data)
                self.hist[k]['counts'] = counts
                self.hist[k]['bin_edges'] = edges
                self.hist[k]['bin_centers'] = (edges[1:] + edges[:-1]) / 2
                x_hist = self.hist[k]['bin_centers']
                y_hist = counts
                mean = sum(x_hist*y_hist)/sum(y_hist)   
                self.hist[k]['fit_params'].add('amp', value=mc_iter/10, min = 0)
                self.hist[k]['fit_params'].add('cen', value = mean) # mean of bin centers
                self.hist[k]['fit_params'].add('wid', value = sum(y_hist*(x_hist-mean)**2)/sum(y_hist)) # sigma of bin centers
                self.hist[k]['x_sim'] = np.linspace(min(x_hist),max(x_hist),1000)
                #Least-square fitting process
                self.hist[k]['fit_result'] = minimize(self.gauss_objective, self.hist[k]['fit_params'], args=(self.hist[k]['bin_centers'], self.hist[k]['counts']))
                self.hist[k]['y_sim'] = self.gaussian(self.hist[k]['x_sim'], *[self.hist[k]['fit_result'].params[w].value for w in self.hist[k]['fit_result'].params])
                
                print(f'\n### Gaussian fit report for {k} ###\n')
                report_fit(self.hist[k]['fit_result'])
            except Exception as e:
                print(f"Error processing histogram for {k}: {e}")
                continue



    def save_monte_carlo_results(self, config_params):
        sample_name = config_params['Sample name']
        model = config_params['Modeling parameters']['Kinetic model']
        output_dir = config_params['Output location']
        monte_carlo_results = {'Parameter':[], 'Opt Value':[], 'Stdev':[]}
        monte_carlo_df = pd.DataFrame(self.monte_carlo_parameters)
        if '2phase' in model:
            all_vars = ['kcat_Dilute', 'KM_Dilute', 'kcat_Dense', 'KM_Dense', 'dGo', 'alpha']
        elif 'Steady state' in model:
            all_vars = ['kcat', 'KM', 'dGo', 'alpha']
        elif "SS" not in model:
            all_vars = ['k1', 'km1', 'k2', 'km2', 'kcat', 'dGo', 'alpha']
        else:
            all_vars = ['kcat', 'KM', 'dGo', 'alpha']

        mc_vars = [k for k in self.monte_carlo_parameters.keys()]
        for k in all_vars:
            if k not in mc_vars:
                if "_Dilute" in k:
                    value = config_params["Modeling parameters"]["Fit parameters"]["Dilute parameters"][k.replace('_Dilute','')]["Value"]
                elif "_Dense" in k:
                    value = config_params["Modeling parameters"]["Fit parameters"]["Dense parameters"][k.replace('_Dense','')]["Value"]
                else:
                    value = config_params["Modeling parameters"]["Fit parameters"][k]["Value"]
                monte_carlo_df[k] = [value]*len(monte_carlo_df)
        
        # Reorder columns to match all_vars order, then add any remaining columns (like 'kobs', 'RSS') at the end
        ordered_cols = [col for col in all_vars if col in monte_carlo_df.columns]
        remaining_cols = [col for col in monte_carlo_df.columns if col not in ordered_cols]
        monte_carlo_df = monte_carlo_df[ordered_cols + remaining_cols]

        kobs = []
        kobs_dilute = []
        kobs_dense = []
        for row in range(len(monte_carlo_df)):
            if "2phase" in model:
                kobs_dilute.append(monte_carlo_df['kcat_Dilute'][row] / monte_carlo_df['KM_Dilute'][row])
                kobs_dense.append(monte_carlo_df['kcat_Dense'][row] / monte_carlo_df['KM_Dense'][row])
            elif model != "Steady state distributive":
                kobs.append(calc_kobs(monte_carlo_df['k1'][row], monte_carlo_df['km1'][row], monte_carlo_df['k2'][row], monte_carlo_df['km2'][row], monte_carlo_df['kcat'][row]))
            else:
                kobs.append(monte_carlo_df['kcat'][row] / monte_carlo_df['KM'][row])
        if "2phase" in model:
            monte_carlo_df['kobs_Dilute'] = kobs_dilute
            monte_carlo_df['kobs_Dense'] = kobs_dense
        else:
            monte_carlo_df['kobs'] = kobs
        monte_carlo_df['RSS'] = self.monte_carlo_rss
        monte_carlo_df.to_csv(f"{output_dir}/{sample_name}_MC_param_values_{self.monte_carlo_iterations}_iters.csv", index=False)
        
        for k in all_vars:
            if k not in mc_vars:
                monte_carlo_results['Parameter'].append(k)
                monte_carlo_results['Opt Value'].append(monte_carlo_df[k].values[0])
                monte_carlo_results['Stdev'].append(0.0)
            else:
                monte_carlo_results['Parameter'].append(k)
                monte_carlo_results['Opt Value'].append(monte_carlo_df[k].values[0])
                monte_carlo_results['Stdev'].append(self.monte_carlo_errors[f"{k} error"])

        if "2phase" in model:
            monte_carlo_results['Parameter'].append('kobs_Dilute')
            monte_carlo_results['Opt Value'].append(np.mean(kobs_dilute))
            monte_carlo_results['Stdev'].append(np.std(kobs_dilute))
            monte_carlo_results['Parameter'].append('kobs_Dense')
            monte_carlo_results['Opt Value'].append(np.mean(kobs_dense))
            monte_carlo_results['Stdev'].append(np.std(kobs_dense))
        else:
            monte_carlo_results['Parameter'].append('kobs')
            monte_carlo_results['Opt Value'].append(np.mean(kobs))
            monte_carlo_results['Stdev'].append(np.std(kobs))
        monte_carlo_results = pd.DataFrame(monte_carlo_results)
        monte_carlo_results.to_csv(f"{output_dir}/{sample_name}_MC_errors_{self.monte_carlo_iterations}_iters.csv", index=False)
        
        # self.plot_monte_carlo_distributions(config_params)


    def plot_monte_carlo_distributions(self, config_params):
        sample_name = config_params['Sample name']
        model = config_params['Modeling parameters']['Kinetic model']
        output_dir = config_params['Output location']

        keys = [k for k in self.monte_carlo_parameters.keys() if not k == 'RSS']
        num_params = len(keys)
        
        fig, axes = plt.subplots(nrows=num_params, ncols=num_params+1, figsize=(num_params*5, num_params*3),
                                 width_ratios=[1]*num_params+[0.2], gridspec_kw={'wspace':0.1})
        gs = axes[0, 0].get_gridspec()
        for i, param1 in enumerate(keys):
            for j, param2 in enumerate(keys):
                if i == j:
                    x_hist = self.hist[param1]['bin_centers']
                    y_hist = self.hist[param1]['counts']
                    axes[i, j].bar(x_hist,y_hist,color='black',width=(max(x_hist)-min(x_hist))/100,label = 'Data')
                    axes[i, j].plot(self.hist[param1]['x_sim'], self.hist[param1]['y_sim'], color='red', linewidth=2,label = f"Gaussian ($\u03C7^{2}_{{red}}$ = {self.hist[param1]['fit_result'].redchi:.2f})")
                    axes[i, j].set_xlabel(f"$log_{{{10}}}$({self.hist[param1]['pretty name']})")
                    axes[i, j].set_ylabel('Count')
                    axes[i, j].legend()
                    if any(k in param1 for k in ['Dilute', 'Dense']):
                        axes[i, j].set_title(fr"{self.hist[param1]['pretty name']} = {(self.opt_params[param1].value):.1e} $\pm$ {(self.monte_carlo_errors[f'{param1} error']):.1e} (1 s.d.) ${self.param_units[param1.split('_')[1]][i]}$")
                    else:
                        axes[i, j].set_title(fr"{self.hist[param1]['pretty name']} = {(self.opt_params[param1].value):.1e} $\pm$ {(self.monte_carlo_errors[f'{param1} error']):.1e} (1 s.d.) ${self.param_units[i]}$")
                else:
                    axes[i, j].scatter(np.log10(self.monte_carlo_parameters[param2]), np.log10(self.monte_carlo_parameters[param1]), c=self.monte_carlo_rss,cmap="coolwarm", alpha=1,s=10)
                    axes[i, j].set_xlabel(f"$log_{{{10}}}$({self.hist[param2]['pretty name']})")
                    axes[i, j].set_ylabel(f"$log_{{{10}}}$({self.hist[param1]['pretty name']})")
        for ax in axes[:,-1]:
            ax.remove()
        cbar_ax = fig.add_subplot(gs[:,-1])  # Define the position of the colorbar
        fig.colorbar(axes[0,1].collections[0], cax=cbar_ax, label='RSS')
        fig.savefig(f"{output_dir}/{sample_name}_MC_parameter_distribitions_{self.monte_carlo_iterations}_iters.pdf", bbox_inches='tight')
        plt.close()

def make_pdf(pdf_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
    return pdf

def calc_kobs(k1,km1,k2,km2,kcat):
    return kcat / (((km2 + kcat) / k2) * (1 + (km1 / k1)))
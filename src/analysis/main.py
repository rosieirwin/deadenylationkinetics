#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors: Robert W. Harkness, Rose M. Irwin
"""
import os
from error_analysis import ErrorAnalysis
from experiment import FretExperiment
from lmfit import Parameters, minimize, report_fit
from models import generate_model_objects, simulate_full_model, calculate_residuals_simulate_best_fit_data
from minimization import objective_wrapper, residuals, sum_of_squared_residuals
import numpy as np

from plotting import PlotHandler
import sys
from utils import load_data, setup_parameters, write_optimal_parameter_csv, write_parameter_RSS_csv

def main():

    ### Get data, set up fit parameters, constants, etc. ###
    config_params, data = load_data(sys.argv[1])
    hybridization_params, initial_guess_params, all_param_units, varied_params, opt_params, opt_param_units = setup_parameters(config_params, Parameters())
    minimizer_params = []
    experiments = []
    kinetic_models = []
    hybridization_models = []

    # Create the output directory if it doesn't exist
    try:
        output_dir = config_params['Output location']
        split_output_dir = output_dir.split('/')
        for i in range(len(split_output_dir)):
            test_dir = '/'.join(split_output_dir[:i+1])
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
    except KeyError:
        print("Output location not specified in config file. Using 'output/' as output directory.")
        output_dir = 'output/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    ### Fit data ###
    if config_params['Modeling parameters']['Fit'] == True:
        min_method = config_params['Modeling parameters']['Minimizer'] # get minimization method

        print("\n***** Fitting experimental data *****\n")
        experiment = FretExperiment(data, hybridization_params, config_params)
        kinetic_model, hybridization_model = generate_model_objects(experiment, config_params['Modeling parameters']['Kinetic model'],config_params)
        experiments.append(experiment)
        kinetic_models.append(kinetic_model)
        hybridization_models.append(hybridization_model)

        minimizer_result = minimize(objective_wrapper, initial_guess_params, method = min_method, 
                                    args=(experiment, kinetic_model, hybridization_model, simulate_full_model, True), max_nfev=config_params['Modeling parameters']['Max function evaluations'],
                                     nan_policy='omit')
        print()
        report_fit(minimizer_result)
        minimizer_params.append(minimizer_result.params)
        
        # Simulate best fit data and plot
        resids, best_kin_models, best_hybr_models = calculate_residuals_simulate_best_fit_data(experiments, minimizer_params, config_params, residuals)
        
        # Calculate RMSD and define error parameters for error analysis (will be used if error analysis is run later on)
        rmsd = np.sqrt(minimizer_result.chisqr/minimizer_result.ndata)
        error_params = minimizer_result.params

        # Save best parameters in .csv
        write_optimal_parameter_csv(minimizer_result.params.valuesdict(), all_param_units, config_params, varied_params, rmsd)

    ### Simulate data with input parameters ###
        # e.g. to check if parameters are reasonable before trying fit
    elif config_params['Modeling parameters']['Fit'] == False:

        print('\n***** Simulating experimental data *****\n')

        # Simulate best fit data and plot
        experiment = FretExperiment(data, hybridization_params, config_params)
        kinetic_model, hybridization_model = generate_model_objects(experiment, config_params['Modeling parameters']['Kinetic model'],config_params)
        experiments.append(experiment)
        kinetic_models.append(kinetic_model)
        hybridization_models.append(hybridization_model)
        minimizer_params.append(initial_guess_params)
        resids, best_kin_models, best_hybr_models = calculate_residuals_simulate_best_fit_data(experiments, minimizer_params, config_params, residuals)
        
        print(f'RSS for simulated data: {sum_of_squared_residuals(resids[0])}')
        write_parameter_RSS_csv(initial_guess_params, output_dir, sum_of_squared_residuals, resids, config_params)

        # Calculate RMSD and define error parameters for error analysis (will be used if error analysis is run later on)
        rmsd = np.sqrt(sum_of_squared_residuals(resids[0])/len(np.concatenate(resids[0], axis=None)))
        error_params = initial_guess_params


    ### Plotting ###
    plot_handler = PlotHandler(experiments, best_kin_models, best_hybr_models, resids, config_params)
    plot_handler.run_plots()


    ### Error analysis ###
    cores = config_params['Modeling parameters']['Error estimation']['Cores']

    # Run Monte Carlo error estimation
    if config_params['Modeling parameters']['Error estimation']['Monte Carlo']['Run'] == True:
        monte_carlo_iterations = config_params['Modeling parameters']['Error estimation']['Monte Carlo']['Iterations']
        export_sim_data = config_params['Modeling parameters']['Error estimation']['Monte Carlo']['Export simulated data']
        error_analyzer = ErrorAnalysis(error_params, varied_params, cores, monte_carlo_iterations, rmsd, None, None, export_sim_data, opt_param_units)
        error_analyzer.monte_carlo_parameter_dictionary()
        error_analyzer.monte_carlo_fits(experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper)
        error_analyzer.monte_carlo_distributions()
        error_analyzer.save_monte_carlo_results(config_params)

    # Run parameter correlation analysis - will check correlations between any varied parameters
    if config_params['Modeling parameters']['Error estimation']['Error surfaces']['Run'] == True:
        range_factor = config_params['Modeling parameters']['Error estimation']['Error surfaces']['Parameter range factor']
        points = config_params['Modeling parameters']['Error estimation']['Error surfaces']['Points']
        error_analyzer = ErrorAnalysis(error_params, varied_params, cores, None, rmsd, range_factor, points, False, opt_param_units)
        error_analyzer.correlation_pairs()
        error_analyzer.parameter_correlation_fits(experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper)
        error_analyzer.save_parameter_correlation_results(config_params)
        error_analyzer.parameter_correlation_surfaces(config_params)


if __name__ == '__main__':
    main()
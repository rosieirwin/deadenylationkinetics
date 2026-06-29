#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors: Robert W. Harkness, Rose M. Irwin
"""

import numpy as np

def objective_wrapper(params, experiment, kinetic_model, hybridization_model, simulate_full_model, print_current_rss=False):
    """
    Wrapper function for the objective function used in minimization.
    Allows for minimization of the residuals for a multi-step calculation:
        - simulate RNA populations with the given parameters,
        - calculate annealing between RNA species and DNA which is directly translated into FRET,
        - calculate residuals between real data and calculated FRET
    """
    kinetic_model, hybridization_model = simulate_full_model(params, kinetic_model, hybridization_model)
    resid = residuals(experiment.fret, hybridization_model.fret)
    concat_resid = np.concatenate(resid, axis=None)
    if print_current_rss == True:
        print(f'Current RSS: {sum_of_squared_residuals(resid)}', end='\r')
        
    return concat_resid

def residuals(ydata, predicted): 
    """Calculate the residuals between the observed and predicted data."""
    resid = []
    for i, v in enumerate(ydata):
        resid.append((ydata[i] - predicted[i])) 
    return resid

def sum_of_squared_residuals(residuals):
    """Calculate the sum of squared residuals."""
    resid = np.concatenate(residuals, axis=None)
    rss = np.sum(np.square(resid))
    return rss

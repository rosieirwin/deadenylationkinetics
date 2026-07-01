#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors: Robert W. Harkness, Rose M. Irwin
"""

from copy import deepcopy
import numpy as np
from scipy import optimize as opt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.integrate import odeint
import re
import pandas as pd
import os


class DistributiveDeadenylation():
    """
    Distributive deadenylation assumes that enzyme falls off RNA strand after catalysis 
    and rebinds product to catalyze again, i.e. the enzyme is fully distributive. Also assumes that 
    enzyme cannot catalyze anything below TA2 in length, i.e. once only 1 A is left, the enzyme cannot
    continue remove bases from the rest of the strand.
    """
    def __init__(self, fret_experiment):
        self.overall_model = "distributive"
        self.time = fret_experiment.time
        self.rna = fret_experiment.rna
        self.enzyme = fret_experiment.enzyme
        self.n = fret_experiment.n
        self.species_list()

    def species_list(self):
        species = ['E*','E'] # Binding incompetent and competent enzyme
        for x in range(1,self.n+1):
            species.append(f"ETA{x}") # RNA bound enzyme
        for x in range(1,self.n+1): # Must use a second loop because species is list not dict so order matters
            species.append(f"TA{x}") # Free RNA
        species.append('A1') # Free AMP arising from deadenylation
        self.species = species

    def setup_concentrations(self):
        self.concentrations = {spec:[] for spec in self.species}

    def initial_concentration_guesses(self, enzyme, rna, k1, km1, n):
        ## Make list of t=0 concentrations of enzyme and substrate.
        ## Should be all free enzyme, all full-length and free RNA substrate
        ## because no binding or cleavage has occurred yet.
        C0 = []
        C0.append((1/(1+(k1/km1))) * enzyme) # E*, initial guess is equilibrium concentration from E* <-> E with no added RNA
        C0.append(((k1/km1)/(1+(k1/km1))) * enzyme) # E
        for x in range(1,n+1): # ETAi, initially no bound complex
            C0.append(0)
        for x in range(1,n+1):
            if x == n: # TAi
                C0.append(rna) # Last element, initially all RNA is max length
            else:
                C0.append(0) # All other lengths are initially zero conc
        C0.append(0) # A1, initially zero
        self.C0 = C0

    @staticmethod
    def relaxation_matrix(C0, k1, km1, k2, km2, kcat, n):
        """
        Relaxation matrix for nuclease activity; assumes that enzyme falls off RNA strand after catalysis 
        and rebinds product to catalyze again, i.e. the enzyme is fully distributive. Also assumes that 
        enzyme cannot catalyze anything below TA2 in length, i.e. once only 1 A is left, the enzyme cannot
        continue remove bases from the rest of the strand. This matrix can be extended for arbitrary length
        RNA, using the parameter n to set the initial RNA length prior to cleavage by the enzyme.

          d/dt  C(t)  =                       R                                   * C(t)

               [E*]     [-k1 km1 0 0 0 0 0 0 0 ...                                ] [E*]
               [E]      [k1 -km1 km2 km2+kcat km2+kcat -k2[E] -k2[E] ... -k2[E] 0 ] [E]
          d/dt [ETA1] = [0 0 -km2 0 0 ... 0 k2[E] 0 0 ...                       0 ] [ETA1]
               [ETA2]   [0 0 0 -km2-kcat 0 0 ... 0 k2[E] 0 0 ...                0 ] [ETA2]
               [ETA3]   [0 0 0 0 -km2-kcat 0 0 ... 0 k2[E] 0 0 ...              0 ] [ETA3]
                                           ...
               [TA1]    [0 0 km2 kcat 0 0 ... 0 -k2[E] 0 0 ...                  0 ] [TA1]
               [TA2]    [0 0 0 km2 kcat 0 0 ... 0 -k2[E] 0 0 ...                0 ] [TA2]
               [TA3]    [0 0 0 0 km2 kcat 0 0 ... 0 -k2[E] 0 0 ...              0 ] [TA3]
                                           ...        
               [TAn]    [0 0 0 0 ... 0 km2 0 0 ...                     0 -k2[E] 0 ] [TAn]
               [A1]     [0 0 0 kcat kcat ... kcat 0 0 ...                       0 ] [A1]   
 
        ## There are a total of 2 * n + 3 species, so for the relaxation matrix R there are 
        ## 2 * n + 3 rows and columns, e.g. 1 E*, 1 E, 1 A1, n ETAi, and n TAi = 2 * n + 3 total species.
        """
        R = []
        R.append([-k1, km1] + [0] * n + [0] * n + [0])  # E*
        R.append([k1, -km1] + [km2] + [km2+kcat] * (n-1) + [-k2 * C0[1]] * n + [0])  # E
        R.append([0] * 2 + [-km2] + [0] * (n - 1) + [k2 * C0[1]] + [0] * (n - 1) + [0])  # ETA1
        for y in range(2, n + 1):  # ETA2 to ETAi
            R.append([0] * (y + 1) + [-km2-kcat] + [0] * (n - y) + [0] * (y - 1) + [k2 * C0[1]] + [0] * (n - y) + [0])
        R.append([0] * 2 + [km2] + [kcat] + [0] * (n - 2) + [-k2 * C0[1]] + [0] * (n - 1) + [0])  # TA1
        for y in range(2, n):  # TA2 to TAn-1
            R.append([0] * (y + 1) + [km2] + [kcat] + [0] * (n - 2) + [-k2 * C0[1]] + [0] * (n - y) + [0])
        R.append([0] * 2 + [0] * (n - 1) + [km2] + [0] * (n - 1) + [-k2 * C0[1]] + [0])  # TAn
        R.append([0] * 2 + [0] + [kcat] * (n - 1) + [0] * n + [0])  # A1

        return R


    def extract_solved_concentrations(self, solver_result, time):
        tmp = [[] for x in solver_result.y]
        for i, v in enumerate(time):
            idx = np.where(solver_result.t == v)[0][0] # Find index of time point in model that matches time point in experiment
            for j,k in enumerate(solver_result.y):
                tmp[j].append(k[idx])
        self.concentrations['E*'].append(tmp[0])
        self.concentrations['E'].append(tmp[1])
        for x in range(1,self.n+1): # Don't have to use two for loops because concentrations is dict not list
            self.concentrations[f"ETA{x}"].append(tmp[x+1]) # x+1 to move past E* and E, ETAi are before TAi
            self.concentrations[f"TA{x}"].append(tmp[x+self.n+1]) # x+n+1 because ETAi and TAi are separated by n indices
        self.concentrations['A1'].append(tmp[-1]) # A1 is last


    def calculate_total_rna_concentrations(self):
        self.total_rna_concentrations = {k:[] for k in self.concentrations.keys() if 'E' not in k}
        for i, v in enumerate(self.enzyme):
            for j in range(1, self.n+1):
                self.total_rna_concentrations[f'TA{j}'].append([sum(x) for x in zip(self.concentrations[f'TA{j}'][i], self.concentrations[f'ETA{j}'][i])]) # TAi,T = [TAi] + [ETAi]
            self.total_rna_concentrations['A1'].append(self.concentrations['A1'][i]) # TAi,T = [TAi] + [ETAi]


    def simulate_kinetics(self, params):
        ## Run numerical integration of rate equations for a given kinetic model from t=0, returns Ci(t)
        ## Needs initial guesses for concentrations of each species at t=0

        k1 = params['k1'].value
        km1 = params['km1'].value
        k2 = params['k2'].value
        km2 = params['km2'].value
        kcat = params['kcat'].value

        self.setup_concentrations()
        # for r, rna in enumerate(self.rna):
        for i, v in enumerate(self.enzyme):
            if len(self.rna) == 1:
                rna = self.rna[0]
            else:
                rna = self.rna[i]
            if self.enzyme[i] == 0: # No enzyme means nothing happens, all RNA is full length at all times
                self.concentrations[f'E*'].append([0 for x in range(len(self.time[i]))])
                self.concentrations[f'E'].append([0 for x in range(len(self.time[i]))])
                self.concentrations[f'A1'].append([0 for x in range(len(self.time[i]))])

                for l in range(1, self.n+1):
                    if l == self.n:
                        self.concentrations[f'TA{self.n}'].append([rna for x in range(len(self.time[i]))])
                        self.concentrations[f'ETA{self.n}'].append([0 for x in range(len(self.time[i]))])
                    else:
                        self.concentrations[f'TA{l}'].append([0 for x in range(len(self.time[i]))])
                        self.concentrations[f'ETA{l}'].append([0 for x in range(len(self.time[i]))])
            else:
                self.initial_concentration_guesses(self.enzyme[i], rna, k1, km1, self.n)
                param_args = {'k1':k1, 'km1':km1, 'k2':k2, 'km2':km2, 'kcat':kcat, 'n':self.n}
                time_span = (np.min(self.time[i]),np.max(self.time[i]))
                initial_concs = self.C0
                rate_func = self.relaxation_matrix
                t_return = np.unique(np.array(self.time[i]))  # only solve for unique time points
                solver_result = solve_ivp(propagator,time_span,initial_concs,t_eval=t_return,method='BDF',first_step=1e-12,rtol=1e-3,atol=1e-19,args=(rate_func, param_args))
                self.extract_solved_concentrations(solver_result,self.time[i])


class TwoPhaseSteadyStateDistributiveDeadenylation():
    """
    Steady-state model for distributive kinetics for two phases

    RNA populations are calculated with the steady-state simplification where KM = (kcat+km2)/k2

    A relaxation matrix is not used in this model (but it could be).
    """
    def __init__(self, fret_experiment):
        self.overall_model = "2phase-ss-dist"
        self.get_phase_values(fret_experiment, fret_experiment.partitioning_data)

        self.time = fret_experiment.time
        self.rna = fret_experiment.rna
        self.enzyme = fret_experiment.enzyme
        self.n = fret_experiment.n
        self.species_list()
        self.separate_rnas = {"Dense":[], "Dilute":[], "Total":[]}
        
    def get_phase_values(self, fret_experiment, partitioning_data):
        ## For each enzyme concentration in the experiment, get the concentration of enzyme in the dense phase
        self.enzyme_dense = []
        self.enzyme_dilute = []
        self.rna_dense = []
        self.rna_dilute = []
        self.total_volume = []
        self.rna_partition_coefficient = []
        self.enz_partition_coefficient = []
        self.enz_dvf = []
        self.rna_dvf = []
        self.rna_total = []
        
        for enz_conc in fret_experiment.enzyme:

            idx = partitioning_data.index[partitioning_data['Total Enzyme'] == enz_conc][0]
            self.rna_total.append(partitioning_data.loc[idx,'Total RNA'])
            self.total_volume.append(partitioning_data.loc[idx,'Total Volume'])
  
            self.rna_partition_coefficient.append(partitioning_data.loc[idx,'RNA PC'])
            self.enz_partition_coefficient.append(partitioning_data.loc[idx,'Enzyme PC'])
            self.enz_dvf.append(partitioning_data.loc[idx,'Enzyme DVF'])
            self.rna_dvf.append(partitioning_data.loc[idx,'RNA DVF'])

            rna_dvf = self.rna_dvf[-1]
            rna_dilVF = 1 - rna_dvf
            enz_dvf = self.enz_dvf[-1]
            enz_dilVF = 1 - self.enz_dvf[-1]

            self.rna_dilute.append(self.rna_total[-1] / (self.rna_partition_coefficient[-1] * rna_dvf + rna_dilVF))
            self.rna_dense.append(self.rna_dilute[-1] * self.rna_partition_coefficient[-1])
            self.enzyme_dilute.append(enz_conc / (self.enz_partition_coefficient[-1] * enz_dvf + enz_dilVF))
            self.enzyme_dense.append(self.enzyme_dilute[-1] * self.enz_partition_coefficient[-1])

    def species_list(self):
        species = []
        species.append('A1')
        for x in range(1,self.n+1):
            species.append(f"TA{x}") # Total RNA, TAi,tot.
        self.species = species

    def simulate_kinetics(self, params):
        self.setup_concentrations() # Must do this to reset concentration dictionary each iteration of the fit
        for ri, rna_total in enumerate(self.rna):
            for ei, enzyme_total in enumerate(self.enzyme):

                kcat = [params[f'kcat_{phase}'].value for phase in ['Dense', 'Dilute']]
                KM = [params[f'KM_{phase}'].value for phase in ['Dense', 'Dilute']]
                
                enzyme = [self.enzyme_dense[ei], self.enzyme_dilute[ei]]
                enz_DVF = self.enz_dvf[ei]
                rna_DVF = self.rna_dvf[ei]
                part_coeff = self.rna_partition_coefficient[ei]
                rna = {}
                rna['Dilute']= rna_total / (rna_DVF * (part_coeff - 1) + 1)
                rna['Dense'] = part_coeff * rna['Dilute']

                if self.enzyme[ei] == 0: # No enzyme means nothing happens, all RNA is full length at all times
                    for phase in ['Dense', 'Dilute']:
                        self.concentrations[phase][f'A1'].append(np.zeros(len(self.time[ei])))
                        for l in range(1, self.n+1):
                            if l == self.n:
                                self.concentrations[phase][f'TA{self.n}'].append([rna[phase] for x in range(len(self.time[ei]))])
                            else:
                                self.concentrations[phase][f'TA{l}'].append([0 for x in range(len(self.time[ei]))])
                else:
                    self.initial_concentration_guesses(rna, self.n)

                    time_span = (np.min(self.time[ei]), np.max(self.time[ei]))
                    t_return = np.unique(np.array(self.time[ei]))  # only solve for unique time points
                    initial_rna = [x for x in (self.initial_concentrations['Dense'] + self.initial_concentrations['Dilute'])]
                    self.separate_rnas = {"Time":[], "Dense":[], "Dilute":[], "Total":[]}

                    _ = solve_ivp(self.rate_wrapper, time_span, initial_rna, t_eval=t_return, 
                                            method='Radau', first_step=1e-12, atol=1e-20, dense_output=True,
                                            args=(part_coeff, rna_DVF, kcat, KM, self.n, enzyme))
                    self.extract_solved_concentrations_from_separate_rnas(self.time[ei])

    def setup_concentrations(self):
        self.concentrations = {spec:[] for spec in self.species}

    def initial_concentration_guesses(self, rna, n):
        self.initial_concentrations = {phase: [[0 for i in range(n)]+[rna[phase]]][0] for phase in ['Dense', 'Dilute']}

    def rate_wrapper(self, t_span, time_rnas, part_coeff, rna_DVF, kcat, KM, n, all_enzyme):
        """        
        Given separate concentrations of RNA in the dilute and dense phases, calculate the total 
        concentration of each RNA species across both phases, then calculate the rates based on 
        the total concentrations. This is necessary because the kinetic model is based on total 
        concentrations, but the ODE solver is solving for separate concentrations in each phase.
        """

        dilVF = 1 - rna_DVF

        rna_dense = list(time_rnas[0:n+1])
        rna_dilute = list(time_rnas[n+1:])

        sum_rnas = [x * rna_DVF + y * dilVF for x,y in zip(rna_dense, rna_dilute)]
        # Separate the total RNA concentrations back into dense and dilute phase concentrations to use as input for the next time point in the ODE solver
        rna_dilute = [Ct / (part_coeff * rna_DVF + dilVF) for Ct in sum_rnas]
        rna_dense = [rden * part_coeff for rden in rna_dilute]
        self.separate_rnas['Time'].append(t_span)
        self.separate_rnas['Dense'].append([float(x) for x in rna_dense])
        self.separate_rnas['Dilute'].append([float(x) for x in rna_dilute])
        self.separate_rnas['Total'].append([float(x) for x in sum_rnas])

        dense_rate = self.rates(t_span, rna_dense, kcat[0], KM[0], n, all_enzyme[0])
        dilute_rate = self.rates(t_span, rna_dilute, kcat[1], KM[1], n, all_enzyme[1])
        
        rate_out = [x for x in dense_rate] + [x for x in dilute_rate]

        return rate_out
    
    def rates(self, t_span, time_rnas, kcat, KM, n, total_enzyme):
        rates = []
        kobs = kcat/KM
        time_rnas = [x if x > 0 else 0 for x in time_rnas] # Set any negative concentrations to zero
        rates.append(kobs * total_enzyme * time_rnas[2]) # TA1 (only depends on i+1)
        for i in range(2, n):
            rates.append(kobs * total_enzyme * (time_rnas[i+1] - time_rnas[i]))
        rates.append(-kobs * total_enzyme * time_rnas[n])
        if time_rnas[n] + rates[-1] < 0: # If the rate of change of TA1 is greater than the concentration of TA1, set the rate to be equal to the concentration of TA1 (i.e., all TA1 is converted to A1 in that time step)
            rates[-1] = -time_rnas[n]
        rates = [np.sum(-x if x < 0 else 0 for x in rates)] + rates # A1, any species that is losing concentration means an equal amount of A1 is produced  

        return rates

    def extract_solved_concentrations_from_separate_rnas(self, time_expt):
        
        """
        solver_result.y is an array of lists where each list corresponds to an RNA species

        Within each list is the concentration of that species at each unique timepoint, meaning if there are two measurements
        i.e., replicates, at a given timepoint, there is only one value in solver_result.y for that time point
        
        The goal of this function is to make the timepoint lists the same length as the real data. This will make calculating
        residuals easier because the solved value lists will be the same length as the real value lists

        Resample solver result for all experimental time points, not just the unique ones. 
        The experimental data may have more than one FRET point per time value, and since the kinetics 
        solver has to return only unique, increasing time points, it is necessary to resample the solver 
        result to generate data with the same dimensions as the experimental input. 

        Finds the index of the time point in the kinetics (solver) result that matches the time point in experiment, 
        then uses this to resample the solver result based on the number of points at each time value 
        in the experimental data.
        """

        for cat in self.separate_rnas:
            if cat != 'Time':
                self.separate_rnas[cat] = np.transpose(self.separate_rnas[cat])

        # Average values for repicated timepoints
        unique_time = np.unique(self.separate_rnas['Time'])
        averaged_separate_rnas = {cat:[[] for species in self.separate_rnas[cat]] for cat in self.separate_rnas if cat != 'Time'}
        for time in unique_time:
            idx = np.where(np.array(self.separate_rnas['Time']) == time)[0]
            for cat in averaged_separate_rnas:
                for species in range(len(self.separate_rnas[cat])):
                    averaged_separate_rnas[cat][species].append(np.mean(np.array(self.separate_rnas[cat][species])[idx], axis=0))
                    if time == 0 and averaged_separate_rnas[cat][species][-1]  < 1e-15: 
                        averaged_separate_rnas[cat][species][-1] = 0 # Set t=0 value to initial concentration guess, because the solver is not returning a value for t=0 and we want to make sure the resampled solver result starts at the correct initial concentration
        # For each cateogry and species, use np.interp to resample the averaged_separate_rnas for the timepoints in time_expt
        interpolated_separate_rnas = {cat:[[] for species in averaged_separate_rnas[cat]] for cat in averaged_separate_rnas}
        for cat in averaged_separate_rnas:
            for species in range(len(averaged_separate_rnas[cat])):
                interpolated_separate_rnas[cat][species] = np.interp(time_expt, unique_time, averaged_separate_rnas[cat][species])              

        # Build the concentrations for FRET in the correct format (keep A1 separate for format of dictionary name)
        self.concentrations['A1'].append(interpolated_separate_rnas['Total'][0])
        for x in range(1, self.n+1):
            self.concentrations[f"TA{x}"].append(interpolated_separate_rnas['Total'][x])

        self.separate_rnas = interpolated_separate_rnas

    def calculate_total_rna_concentrations(self):
        """
        No need to sum over free and enzyme-bound concentrations to get the total 
        concentration of each RNA species in this model, 
        because the kinetic simulations already output the total concentrations
        """
        self.total_rna_concentrations = deepcopy(self.concentrations) 

    # def write_solved_concentrations(self, t_return, solver_result, ei):
    #     with open(f'output_ss/rna_pops/solver_result_{ei}.csv', 'w') as f:
    #         f.write(f"Time points:\n")
    #         f.write(f"{",".join(solver_result.t.flatten().astype(str))}\n")
    #         f.write(f"Concentrations:\n")
    #         for conc in solver_result.y:
    #             f.write(f"{",".join(conc.flatten().astype(str))}\n")

    #     for cat in self.separate_rnas:
    #         if cat != 'Time':
    #             self.separate_rnas[cat] = np.transpose(self.separate_rnas[cat])

    #     with open(f'output_ss/rna_pops/separate_rnas_{ei}.csv', 'w') as f:
    #         f.write(f"Time points:\n")
    #         f.write(f"{",".join(t_return.flatten().astype(str))}\n")
    #         for cat in self.separate_rnas:
    #             f.write(f"\n{cat}\n")
    #             if cat != "Time":
    #                 for row in self.separate_rnas[cat]:
    #                     f.write(f"{",".join(row.flatten().astype(str))}\n")
    #             else:
    #                 f.write(",".join([str(x) for x in self.separate_rnas[cat]]))


class ProcessiveDeadenylation():
    """
    Processive deadenylation assumes that enzyme only falls off RNA strand after
    catalysis to TA1, i.e. the enzyme is fully processive. Also assumes that 
    enzyme cannot catalyze anything below TA2 in length, i.e. once only 1 A is left, 
    the enzyme cannot continue remove bases from the rest of the strand.
    """
    def __init__(self, fret_experiment):
        self.overall_model = "processive"
        self.time = fret_experiment.time
        self.rna = fret_experiment.rna
        self.enzyme = fret_experiment.enzyme
        self.n = fret_experiment.n
        self.species_list()


    def species_list(self):
        species = ['E*','E'] # Binding incompetent and competent enzyme
        for x in range(2,self.n+1):
            species.append(f"ETA{x}") # RNA bound enzyme            
        species.append('TA1') # Free AMP arising from deadenylation
        species.append('A1') # Free AMP arising from deadenylation
        species.append(f"TA{self.n}") # Free RNA
        self.species = species


    def setup_concentrations(self):
        self.concentrations = {}
        for spec in self.species:
            self.concentrations[spec] = []


    def initial_concentration_guesses(self, enzyme, rna, k1, km1, n):
        ## Make list of t=0 concentrations of enzyme and substrate.
        ## Should be all free enzyme, all full-length and free RNA substrate
        ## because no binding or cleavage has occurred yet.

        C0 = []
        C0.append((1/(1+(k1/km1))) * enzyme) # E*, initial guess is equilibrium concentration
        C0.append(((k1/km1)/(1+(k1/km1))) * enzyme) # E
        for x in range(2,n+1): # ETAi, initially no bound complex
            C0.append(0)
        C0.append(0) # TA1
        C0.append(0) # A1
        C0.append(rna) # TAn, all substrate initially max length

        self.C0 = C0


    @staticmethod
    def relaxation_matrix(C0, k1, km1, k2, km2, kcat, n):
        """
        Relaxation matrix for nuclease activity; assumes that enzyme only falls off RNA strand after
        catalysis to TA1, i.e. the enzyme is fully processive. Also assumes that enzyme cannot catalyze  
        anything below TA2 in length, i.e. once only 1 A is left, the enzyme cannot continue remove bases 
        from the rest of the strand. This matrix can be extended for arbitrary length RNA, using the 
        parameter n to set the initial RNA length prior to cleavage by the enzyme.

         d/dt  C(t)  =                       R                               * C(t)

              [E*]     [-k1 km1 0 0 0 0 0 0 0 ...                          0 ] [E*]
              [E]      [k1 -km1 kcat 0 0 0 ...              0 km2 0 0 -k2[E] ] [E]
         d/dt [ETA2]   [0 0 -kcat kcat 0 0 0 ...                           0 ] [ETA2]
              [ETA3]   [0 0 0 -kcat kcat 0 0 0 ...                         0 ] [ETA3]
                                           ...
              [ETAn]   [0 0 0 0 0 ...                  0 -kcat-km2 0 0 k2[E] ] [ETAn]
              [TA1]    [0 0 kcat 0 0 0 ...                                 0 ] [TA1]
              [A1]     [0 0 kcat kcat kcat kcat ...               kcat 0 0 0 ] [A1]   
              [TAn]    [0 0 0 0 0 0 0 0 0 ...               0 km2 0 0 -k2[E] ] [TAn]

        There are a total of n + 4 species, so for the relaxation matrix R there are
        n + 4 rows and columns, e.g. 1 E*, 1 E, n-1 ETAi, 1 TAn, 1 TA1, A1 = n + 4 total species.
        """
        R = []
        R.append([-k1, km1] + [0] * (n - 1) + [0] + [0] + [0]) # E*
        R.append([k1, -km1] + [kcat] + [0] * (n-3) + [km2] + [0] + [0] + [-k2 * C0[1]]) # E
        for y in range(2, n):  # ETA2 to ETAn-1
            R.append([0] * 2 + [0] * (y - 2) + [-kcat] + [kcat] + [0] * (n - y - 1) + [0] + [0] + [0])        
        R.append([0] * 2 + [0] * (n - 2) + [-kcat - km2] + [0] + [0] + [k2 * C0[1]]) # ETAn, separated from ETA2-ETAn-1      
        R.append([0] * 2 + [kcat] + [0] * (n - 2) + [0] + [0] + [0])  # TA1                
        R.append([0] * 2 + [kcat] * (n - 1) + [0] + [0] + [0])  # A1
        R.append([0] * 2 + [0] * (n - 2) + [km2] + [0] + [0] + [-k2 * C0[1]])  # TAn

        return R


    def extract_solved_concentrations(self, solver_result, time):
        tmp = [[] for x in solver_result.y]
        for i, v in enumerate(time):
            idx = np.where(solver_result.t == v)[0][0] # Find index of time point in model that matches time point in experiment
            for j,k in enumerate(solver_result.y):
                tmp[j].append(k[idx])
        self.concentrations['E*'].append(tmp[0])
        self.concentrations['E'].append(tmp[1])
        for x in range(2,self.n+1): # Don't have to use two for loops because concentrations is dict not list
            self.concentrations[f"ETA{x}"].append(tmp[x])
        self.concentrations[f"TA1"].append(tmp[-3])
        self.concentrations['A1'].append(tmp[-2])
        self.concentrations[f"TA{self.n}"].append(tmp[-1])


    def calculate_total_rna_concentrations(self):
        temp_keys = []
        for k in self.concentrations.keys():
            if 'A' in k:
                temp_keys.append(re.findall(r'[TA0-9]+', k)[0])
        temp_keys = sorted(list(set(temp_keys)))
        self.total_rna_concentrations = {k:[] for k in temp_keys}
        for i, v in enumerate(self.enzyme):
            points = len(self.time[i])
            for j in range(1, self.n+1): # TAi,T = [TAi] + [ETAi] 
                TA_conc = [0] * points
                ETA_conc = [0] * points
                if f"TA{j}" in self.concentrations.keys():
                    TA_conc = self.concentrations[f"TA{j}"][i]
                if f"ETA{j}" in self.concentrations.keys():
                    ETA_conc = self.concentrations[f"ETA{j}"][i]
                self.total_rna_concentrations[f'TA{j}'].append([sum(x) for x in zip(TA_conc, ETA_conc)]) # TAi,T = [TAi] + [ETAi]
            self.total_rna_concentrations['A1'].append(self.concentrations['A1'][i]) # TAi,T = [TAi] + [ETAi]


    def simulate_kinetics(self, params):
        ## Run numerical integration of rate equations for a given kinetic model from t=0, returns Ci(t)
        ## Needs initial guesses for concentrations of each species at t=0

        k1 = params['k1'].value
        km1 = params['km1'].value
        k2 = params['k2'].value
        km2 = params['km2'].value
        kcat = params['kcat'].value

        self.setup_concentrations()
        for i, v in enumerate(self.enzyme):
            if len(self.rna) == 1:
                rna = self.rna[0]
            else:
                rna = self.rna[i]
            if self.enzyme[i] == 0: # No enzyme means nothing happens, all RNA is full length at all times
                self.concentrations[f'E*'].append([0 for x in range(len(self.time[i]))])
                self.concentrations[f'E'].append([0 for x in range(len(self.time[i]))])
                self.concentrations[f'A1'].append([0 for x in range(len(self.time[i]))])
                self.concentrations[f'TA1'].append([0 for x in range(len(self.time[i]))])

                for l in range(2, self.n+1):
                    if l == self.n:
                        self.concentrations[f'TA{self.n}'].append([rna for x in range(len(self.time[i]))])
                        self.concentrations[f'ETA{self.n}'].append([0 for x in range(len(self.time[i]))])
                    else:
                        self.concentrations[f'ETA{l}'].append([0 for x in range(len(self.time[i]))])
            else:
                self.initial_concentration_guesses(self.enzyme[i], rna, k1, km1, self.n)
                param_args = {'k1':k1, 'km1':km1, 'k2':k2, 'km2':km2, 'kcat':kcat, 'n':self.n}
                time_span = (np.min(self.time[i]),np.max(self.time[i]))
                initial_concs = self.C0
                rate_func = self.relaxation_matrix
                t_return = np.unique(np.array(self.time[i]))  # only solve for unique time points
                solver_result = solve_ivp(propagator,time_span,initial_concs,t_eval=t_return,method='BDF',first_step=1e-12,atol=1e-12,args=(rate_func, param_args))
                self.extract_solved_concentrations(solver_result,self.time[i])

   
class SteadyStateDistributiveDeadenylation:
    """
    Steady-state model for distributive kinetics.

    RNA populations are calculated with the steady-state simplification where KM = (kcat+km2)/k2

    A relaxation matrix is not necessary for this model.
    """
    def __init__(self, fret_experiment):
        self.overall_model = "steady_state_distributive"
        self.time = fret_experiment.time
        self.rna = fret_experiment.rna
        self.enzyme = fret_experiment.enzyme
        self.n = fret_experiment.n
        self.species_list()

    def species_list(self):
        species = []
        species.append('A1')
        for x in range(1,self.n+1):
            species.append(f"TA{x}") # Total RNA, TAi,tot.
        self.species = species

    def setup_concentrations(self):
        self.concentrations = {spec:[] for spec in self.species}

    def initial_concentration_guesses(self, rna, n):
        initial_concentrations = [0 for i in range(n+1)]
        initial_concentrations[-1] = rna
        self.initial_concentrations = initial_concentrations
    
    def extract_solved_concentrations(self, solver_result, current_time):
        
        """
        solver_result.y is an array of lists where each list corresponds to an RNA species

        Within each list is the concentration of that species at each unique timepoint, meaning if there are two measurements
        i.e., replicates, at a given timepoint, there is only one value in solver_result.y for that time point
        
        The goal of this function is to make the timepoint lists the same length as the real data. This will make calculating
        residuals easier because the solved value lists will be the same length as the real value lists

        Resample solver result for all experimental time points, not just the unique ones. 
        The experimental data may have more than one FRET point per time value, and since the kinetics 
        solver has to return only unique, increasing time points, it is necessary to resample the solver 
        result to generate data with the same dimensions as the experimental input. 

        Finds the index of the time point in the kinetics (solver) result that matches the time point in experiment, 
        then uses this to resample the solver result based on the number of points at each time value 
        in the experimental data.
        """

        tmp = [[] for x in solver_result.y] # length of number of species
        for ti, time in enumerate(current_time): 
            idx = np.where(solver_result.t == time)[0][0]
            for ci, species_concentration in enumerate(solver_result.y):
                tmp[ci].append(species_concentration[idx])

        # Build the concentrations for FRET in the correct format (keep A1 separate for format of dictionary name)
        self.concentrations['A1'].append(tmp[0])
        for x in range(1, self.n+1):
            self.concentrations[f"TA{x}"].append(tmp[x])

    def calculate_total_rna_concentrations(self):
        """
        No need to sum over free and enzyme-bound concentrations to get the total 
        concentration of each RNA species in this model, 
        because the kinetic simulations already output the total concentrations
        """
        self.total_rna_concentrations = deepcopy(self.concentrations) 

    @staticmethod
    def rates(t_span, total_rnas, kcat, KM, n, total_enzyme):
        rates = []
        kobs = kcat/KM

        rates.append(kobs * total_enzyme * total_rnas[2])
        for i in range(2, n):
            rates.append(kobs * total_enzyme * (total_rnas[i+1] - total_rnas[i]))
        rates.append(-kobs * total_enzyme * total_rnas[n])
        rates = [np.sum(np.abs(rates))] + rates
        return rates
    
    def simulate_kinetics(self, params):

            kcat = params['kcat'].value
            KM = params['KM'].value

            self.setup_concentrations() # Must do this to reset concentration dictionary each iteration of the fit
            for ri, rna in enumerate(self.rna):
                for ei, enzyme in enumerate(self.enzyme):
                    if self.enzyme[ei] == 0: # No enzyme means nothing happens, all RNA is full length at all times
                        self.concentrations[f'A1'].append(np.zeros(len(self.time[ei])))
                        for l in range(1, self.n+1):
                            if l == self.n:
                                self.concentrations[f'TA{self.n}'].append([rna for x in range(len(self.time[ei]))])
                            else:
                                self.concentrations[f'TA{l}'].append([0 for x in range(len(self.time[ei]))])
                    else:
                        self.initial_concentration_guesses(rna, self.n)
                        time_span = (np.min(self.time[ei]), np.max(self.time[ei]))
                        t_return = np.unique(np.array(self.time[ei]))  # only solve for unique time points
                        solver_result = solve_ivp(self.rates, time_span, self.initial_concentrations, t_eval=t_return, method='BDF',first_step=1e-12, atol=1e-12, args=(kcat, KM, self.n, enzyme))
                        self.extract_solved_concentrations(solver_result, self.time[ei])


class DuplexHybridization:
    """
    This class models the hybridization of RNA with a DNA quencher strand. It translates
    RNA-DNA annealing into FRET values to be fit to the data.
    """
    def __init__(self, fret_experiment):

        self.fit_model = fret_experiment.fit_model
        self.experimental_fret = fret_experiment.fret # Needed for solving baseline params with Ax = B
        self.dGo = fret_experiment.dGo
        self.alpha = fret_experiment.alpha
        self.n = fret_experiment.n
        self.temperature = fret_experiment.temperature
        self.time = fret_experiment.time
        self.QT = fret_experiment.QT
        self.enzyme = fret_experiment.enzyme
        self.rna = fret_experiment.rna

        self.species_list()
        self.initial_concentration_guesses()
        self.calculate_kq()

    def species_list(self):
        species = []
        for x in range(1,self.n+1):
            species.append(f'TA{x}') # Free RNA
        for x in range(1,self.n+1):
            species.append(f'TA{x}Q') # RNA hybridized to DNA quencher strand
        species.append('Q') # Free quencher strand
        self.species = species

    def setup_concentrations(self):
        self.concentrations = {}
        for spec in self.species:
            self.concentrations[spec] = [[] for concentration in self.enzyme]
        self.annealed_fraction  =[[] for x in self.enzyme]

    def initial_concentration_guesses(self):
        self.C0 = []
        rna = self.rna[0]
        for x in range(self.n):
            self.C0.append(rna) # [TAi], free RNA
        for x in range(self.n):
            self.C0.append(rna) # [TAiQ], RNA annealed to DNA quencher strand
        self.C0.append(self.QT) # [Q], free DNA quencher

    def calculate_kq(self):
        i = np.array([I for I in np.arange(1,self.n+1)])
        dG = self.dGo + self.alpha * i # dG for forming hybrid RNA:DNA duplex as function of RNA length
        R = 8.3145e-3 # units of kJ/mol for dG, change to 1.987e-3 if you like kcal/mol but then also need to change dGo and alpha inputs to kcal/mol
        self.KQ = np.exp(-dG/(R * self.temperature))

    def extract_solved_concentrations(self, solver_result, ei):
        for index, value in enumerate(self.concentrations.keys()):
            self.concentrations[value][ei].append(solver_result.x[index])

    def simulate_hybridization(self, kinetic_model):
    ## Solve for concentrations of free and hybridized RNA after stopping reaction and adding quencher DNA strand
    ## Needs initial guesses for concentrations as in the kinetic part
        self.setup_concentrations()
        for ei, enzyme in enumerate(kinetic_model.enzyme):
            for ti, time in enumerate(kinetic_model.time[ei]):
                self.get_total_rna_concentrations({k:kinetic_model.concentrations[k][ei][ti] for k in kinetic_model.concentrations.keys()})
                solver_result = root(self.hybrid_duplex_equations, self.C0, args=(self.n, self.QT, self.total_concentrations, self.KQ), method='hybr')
                self.extract_solved_concentrations(solver_result, ei) # Need enzyme index to extend concentration list for each enzyme concentration
                self.annealed_fraction[ei].append(np.sum([self.concentrations[k][ei][ti]/self.rna for k in self.concentrations if ('Q' in k) & (k[0] != 'Q')])) # Want everything annealed to Q, i.e. TAiQ, but not free Q

    def get_total_rna_concentrations(self, prior_to_hybridization_concentrations):
        if any(self.fit_model == x for x in ['Steady state distributive', '2phase steady state distributive', '2phase SS distributive']):
            self.total_concentrations = [prior_to_hybridization_concentrations[f'TA{x}'] for x in range(1, self.n+1)] # TAi,T = [TAi] + [ETAi]
        else:
            concs = []
            for x in range(1,self.n+1):
                TA_conc = 0
                ETA_conc = 0
                if f'TA{x}' in prior_to_hybridization_concentrations.keys():
                    TA_conc = prior_to_hybridization_concentrations[f'TA{x}']
                if f'ETA{x}' in prior_to_hybridization_concentrations.keys():
                    ETA_conc = prior_to_hybridization_concentrations[f'ETA{x}']
                concs.append(TA_conc + ETA_conc)
            self.total_concentrations = concs

    @staticmethod
    def hybrid_duplex_equations(C0, n, QT, TAiT, KQ):
        """
        This takes the concentrations of each RNA species at each time point and calculates how much
        of each becomes annealed to the capture strand Q according to the affinity constant KQ.
        Essentially, this calculates the concentrations of a series of hybrid duplexes over time,
        once the deadenylation reaction is quenched and capture strand is added to probe the FRET value.
        """
        eqs = []
        eqs.append(-QT + np.sum(C0[n:])) # 0 = -QT + [Q] + [TA1Q] + ... + [TAnQ], mass conservation DNA quencher
        for x in range(n):
            eqs.append(-TAiT[x] + C0[x] + C0[x+n]) # 0 = -TAiT + [TAi] + [TAiQ], mass conservation each RNA length
        for x in range(n):
            eqs.append(KQ[x] * C0[x] * C0[-1] - C0[x+n]) # KQi * [TAi] * [Q] - [TAiQ] = 0, affinity constant of each hybrid duplex
        return eqs

    def calculate_fret(self):
        for ei, enzyme in enumerate(self.enzyme):
            annealed = np.array(self.annealed_fraction[ei])
            if len(annealed) == 0:
                scaled = []
            else:
                scaled = annealed / annealed[0] if annealed[0] != 0 else annealed
                scaled[0] = 1.0
            self.annealed_fraction[ei] = scaled.tolist()
        self.fret = deepcopy(self.annealed_fraction)


def propagator(t, C, func, constants): # Used in scipy.integrate.solve_ivp, general propagation function for use by kinetic model objects
    R = func(C, **constants) # Make relaxation matrix
    return np.matmul(R,C) # Calculates concentration fluxes, d/dt C


def generate_model_objects(fret_experiment, fit_model, config_params):
    fret_experiment.fit_model = fit_model
    if fit_model == 'Distributive':
        kinetic_model = DistributiveDeadenylation(fret_experiment)
    if fit_model == 'Processive':
        kinetic_model = ProcessiveDeadenylation(fret_experiment)
    if fit_model == '2phase steady state distributive' or fit_model == '2phase SS distributive':
        kinetic_model = TwoPhaseSteadyStateDistributiveDeadenylation(fret_experiment)
    if fit_model == 'Steady state distributive':
        kinetic_model = SteadyStateDistributiveDeadenylation(fret_experiment)

    hybridization_model = DuplexHybridization(fret_experiment)

    return kinetic_model, hybridization_model


def simulate_full_model(params, kinetic_model, hybridization_model):
    kinetic_model.simulate_kinetics(params)
    kinetic_model.calculate_total_rna_concentrations()
    hybridization_model.simulate_hybridization(kinetic_model)
    hybridization_model.calculate_fret()
    return kinetic_model, hybridization_model


def calculate_residuals_simulate_best_fit_data(fret_expts, opt_params, config_params, residuals):
    # Calculate fit residuals and simulate data over finely sampled experimental temperature range to generate smooth best fit data and population plots
    best_kin_models = []
    best_hybr_models = []
    resid = []
    for i, fret_expt in enumerate(fret_expts):
        kinetic_model, hybridization_model = generate_model_objects(fret_expt, config_params['Modeling parameters']['Kinetic model'], config_params)
        kinetic_model, hybridization_model = simulate_full_model(opt_params[i], kinetic_model, hybridization_model)
        resid.append(residuals(fret_expt.fret, hybridization_model.fret))

        max_time = max(np.array([max(time_vector) for time_vector in fret_expt.time])) # Re-simulate with finely sampled experimental time vectors
        sim_time = [np.linspace(0, max_time, 300) for time_vector in fret_expt.time]
        sim_fret_expt = deepcopy(fret_expt)
        sim_fret_expt.time = sim_time
        sim_kinetic_model, sim_hybridization_model = generate_model_objects(sim_fret_expt, config_params['Modeling parameters']['Kinetic model'], config_params)
        sim_kinetic_model.simulate_kinetics(opt_params[i])
        sim_kinetic_model.calculate_total_rna_concentrations()
        sim_hybridization_model.simulate_hybridization(sim_kinetic_model)
        sim_hybridization_model.calculate_fret()    

        best_kin_models.append(sim_kinetic_model)
        best_hybr_models.append(sim_hybridization_model)

        if config_params["Modeling parameters"]["Kinetic model"] == "2phase":
            ##replace concentrations with partitioned concentrations
            pass

        if config_params['Modeling parameters']['Export simulated data'] == True:
            export_best_fit_data(fret_expt, max_time, config_params, opt_params[i], i)                
        
    return resid, best_kin_models, best_hybr_models


def export_best_fit_data(fret_expt, max_time, config_params, opt_params, i):
    best_fit_path = 'best_fit_data/'
    sim_fret_expt = deepcopy(fret_expt)
    sim_time = [np.linspace(0, max_time, 20) for time_vector in fret_expt.time]
    sim_fret_expt.time = sim_time
    sim_kinetic_model, sim_hybridization_model = generate_model_objects(sim_fret_expt, config_params['Modeling parameters']['Kinetic model'], config_params)
    sim_kinetic_model.simulate_kinetics(opt_params)
    sim_kinetic_model.calculate_total_rna_concentrations()
    sim_hybridization_model.simulate_hybridization(sim_kinetic_model)
    sim_hybridization_model.calculate_fret()
    

    if not os.path.exists(best_fit_path):
        os.makedirs(best_fit_path)
    export_data = []
    for vi, v in enumerate(sim_hybridization_model.enzyme):
        enzyme_conc = [v] * len(sim_time[vi])
        rna_conc = [sim_hybridization_model.rna[0]] * len(sim_time[vi])
        dna_conc = [sim_hybridization_model.QT] * len(sim_time[vi])
        export_data.append(pd.DataFrame(np.array([sim_time[vi], sim_hybridization_model.fret[vi], enzyme_conc, rna_conc, dna_conc]).T, columns=["Time", "FRET", "Enzyme", "RNA", "DNA"]))
    export_data = pd.concat(export_data)
    export_data.reset_index(drop=True, inplace=True)
    export_data.to_csv(f"{best_fit_path}{config_params['Sample name']}_best-fit_{i+1}.csv", index=False)
    
    if not os.path.exists('output/E_populations/'):
        os.makedirs('output/E_populations/')
    export_E = []
    for vi, v in enumerate(sim_kinetic_model.enzyme):
        enzyme_conc = [v] * len(sim_time[vi])
        E_conc = sim_kinetic_model.concentrations['E'][vi]
        Estar_conc = sim_kinetic_model.concentrations['E*'][vi]
        for x,y,z,g in zip(sim_time[vi], enzyme_conc, E_conc, Estar_conc):
            export_E.append([x,y,z,g])
    export_E = pd.DataFrame(export_E, columns=["Time", "Enzyme", "E", "E*"])
    export_E.to_csv(f"output/E_populations/{config_params['Sample name']}_E_populations.csv", index=False)

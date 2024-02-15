"""
Worked example of performing a basic state transfer optimisation for a given
process duration using the adjoint method
"""

import numpy as np
import oqupy
from oqupy.process_tensor import BaseProcessTensor
import matplotlib.pyplot as plt
from oqupy.helpers import get_full_timesteps,get_half_timesteps
from scipy.linalg import expm
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import sys

initial_state = oqupy.operators.spin_dm('x-')
target_state = oqupy.operators.spin_dm('x+')
# generate a pt using gradient tutorial and copy it to current directory
process_tensor_type = 'default'

initial_guess_x = 0.0
initial_guess_z = np.pi / 5

bounds_x = (-5.0,5.0)
bounds_z = (-1.0,1.0)

if process_tensor_type == 'default':
    # 5ps pt, super-ohmic sd. dt=0.05,dkmax=60,epsrel=10^-7
    process_tensor = oqupy.import_process_tensor(
                'optimisation_pt.processTensor','simple')

half_timestep_times = get_half_timesteps(process_tensor,0)

def get_hamiltonian(hx:np.ndarray,hz:np.ndarray,pt:BaseProcessTensor):
    """
    Returns a callable which takes a single parameter, t, and returns the
    hamiltonian of the two level system at that time. This function takes a
    process tensor, and the magnitude of $h_z$ and $h_x$ at each of those
    timesteps
    """

    # expval times including endtime, to generate the last "pixel"
    expval_times_p1 = get_full_timesteps(pt,0,inc_endtime=True)
    assert hx.size == expval_times_p1.size-1, \
        'hx must be same length as number of timesteps, without endtime'
    assert hz.size == expval_times_p1.size-1, \
        'hz must be same length as number of timesteps, without endtime'

    # duplicate last element so any time between t_f-dt and t_f falls within
    # this 'pixel' otherwise scipy interp1d doesn't like extrapolating so calls
    # it out of bounds
    hx_p1 = np.concatenate((hx,np.array([hx[-1]])))
    hz_p1 = np.concatenate((hz,np.array([hz[-1]])))

    hx_interp = interp1d(expval_times_p1,hx_p1,kind='zero')
    hz_interp = interp1d(expval_times_p1,hz_p1,kind='zero')

    def hamiltonian_t(t):
        _hx = hx_interp(t)
        _hz = hz_interp(t)

        hx_sx = 0.5 * oqupy.operators.sigma('x') * _hx
        hz_sz = 0.5 * oqupy.operators.sigma('z') * _hz
        hamiltonian = hz_sz + hx_sx
        return hamiltonian

    return hamiltonian_t

def dpropagator(hamiltonian,
    t, # expectation value times
    dt,
    op,
    h):
    """
    deriv of propagator wrt either a pre node or a post node
    """

    liouvillian_plus_h=-1j * oqupy.operators.commutator(hamiltonian(t)+h*op)
    liouvillian_minus_h=-1j * oqupy.operators.commutator(hamiltonian(t)-h*op)

    propagator_plus_h=expm(liouvillian_plus_h*dt/2.0).T
    propagator_minus_h=expm(liouvillian_minus_h*dt/2.0).T

    deriv=(propagator_plus_h-propagator_minus_h)/(2.0*h)
    return deriv

def sum_adjacent_elements(array:np.ndarray)-> np.ndarray:
    """
    Takes an array where the length is even and summs the two adjacent elements.
    """
    # maybe this goes in helpers.py or utils.py?
    half_the_size = array.size / 2
    assert (half_the_size).is_integer(), \
        'if one output from both pre and post node is given, result must be even'
    half_the_size = int(half_the_size)

    # https://stackoverflow.com/a/29392016
    summed_array = array.reshape((half_the_size,2)).sum(axis=1)
    return summed_array

def cost_function(control_parameters,
                  pt):
    hamiltonian_t = get_hamiltonian(
                                    hx=control_parameters[:len(pt)], # first half is sigma x
                                    hz=control_parameters[len(pt):], # second half is sigma z
                                    pt=pt)
    system = oqupy.TimeDependentSystem(hamiltonian_t)
    dynamics = oqupy.compute_dynamics(system=system,
                        initial_state=initial_state,
                        process_tensor=pt,
                        record_all=False, # only record final states
                        progress_type='silent')
    final_state = dynamics.states[-1]
    infidelity = 1 - np.matmul(final_state,target_state).trace()
    return infidelity.real

def gradient_function(control_parameters,
                      pt):
    hamiltonian_t = get_hamiltonian(
                                    hx=control_parameters[:len(pt)], # first half is sigma x
                                    hz=control_parameters[len(pt):], # second half is sigma z
                                    pt=pt)
    system = oqupy.TimeDependentSystem(hamiltonian_t)
    # list of the derivs of the propagators w.r.t. the control parameters
    dprop_dpram_derivs_x = []
    dprop_dpram_derivs_z = []

    for i in range(half_timestep_times.size):
        dprop_x = dpropagator(
                            hamiltonian_t,
                            half_timestep_times[i],
                            process_tensor.dt,
                            op=0.5*oqupy.operators.sigma('x'),
                            h = 10**(-6))
        dprop_dpram_derivs_x.append(dprop_x)

        dprop_z = dpropagator(
                            hamiltonian_t,
                            half_timestep_times[i],
                            process_tensor.dt,
                            op=0.5*oqupy.operators.sigma('z'),
                            h = 10**(-6))
        dprop_dpram_derivs_z.append(dprop_z)

    gradient_with_x = oqupy.gradient(system=system,
                                     process_tensor=pt,
                                    initial_state=initial_state,
                                    target_state=target_state,
                                    dprop_dparam_list=dprop_dpram_derivs_x,
                                    progress_type='silent')
    # extract derivs of control parameters over half a step, so will need to sum
    # them later to get the full step
    total_derivs_x = gradient_with_x.total_derivs

    # since we have already done forwardprop and backprop, we can reuse result
    # by passing the object returned by oqupy.gradient back into oqupy.gradient,
    # except this time with a different dprop_dparam list.
    # NOTE that now we don't need to specify initial and target state as the
    # forward and backprop have already been done
    gradient_with_z = oqupy.gradient(system=system,
                                     process_tensor=pt,
                                     gradient_dynamics=gradient_with_x,
                                     dprop_dparam_list=dprop_dpram_derivs_z)
    # extract derivs of CPs over half steps
    total_derivs_z = gradient_with_z.total_derivs

    # combine two results into one array that's the same shape as the control
    # parameters, for L-BFGS-B algorithm
    total_derivs = np.concatenate((total_derivs_x,total_derivs_z))
    total_derivs = -1 * total_derivs.real # get infidelity
    # sum adjacent elements to get derivs over whole timesteps
    total_derivs_summed = sum_adjacent_elements(total_derivs)
    # L-BFGS-B needs to have a jacobian expressed as a fortran type array
    # (column major)
    total_derivs_fortran = np.asfortranarray(total_derivs_summed)
    return total_derivs_fortran

initial_guess = np.zeros(2*len(process_tensor))
initial_guess[:len(process_tensor)] = initial_guess_x
initial_guess[len(process_tensor):] = initial_guess_z

lower_bound = np.zeros(2*len(process_tensor))
lower_bound[:len(process_tensor)] = bounds_x[0]
lower_bound[len(process_tensor):] = bounds_z[0]

upper_bound = np.zeros(2*len(process_tensor))
upper_bound[:len(process_tensor)] = bounds_x[1]
upper_bound[len(process_tensor):] = bounds_z[1]

bounds_instance = Bounds(lb=lower_bound,ub=upper_bound)
# plt.plot(initial_guess,color='black')
# plt.plot(lower_bound)
# plt.plot(upper_bound)
# plt.show()
result = minimize(cost_function,
                  initial_guess,
                  method='BFGS',
                  jac=gradient_function,
                  # bounds=bounds_instance,
                  args=(process_tensor),
                  options={'disp': True})

np.save('result',result.x)
plt.plot(result.x)
plt.show()







import numpy as np
import matplotlib.pyplot as plt
from oqupy.helpers import get_full_timesteps
import oqupy
from optimisation_example import get_hamiltonian,initial_state,target_state,cost_function

plot_dynamics = True
plot_control_parameters = True

control_parameter_results = np.load('result.npy')

size = int(control_parameter_results.size / 2)

pt = oqupy.import_process_tensor(
            'optimisation_pt.processTensor','simple')
times = get_full_timesteps(pt,0)
hx = control_parameter_results[:size]
hz = control_parameter_results[size:]

if plot_control_parameters:
    plt.figure(1)
    plt.plot(times,hx,label='hx')
    plt.plot(times,hz,label='hz')
    plt.xlabel('t [ps]')
    plt.legend()
# plt.show()

def get_dynamics(pt:oqupy.SimpleProcessTensor,
                 control_parameters:np.ndarray):
    '''
    Takes process tensor and control parameters, as a single list, parses the
    control parameters and returns a oqupy.dynamics object
    '''
    assert 2*len(pt) == control_parameters.size, 'something broke'

    hx = control_parameters[:len(pt)]
    hz = control_parameters[len(pt):]

    ham = get_hamiltonian(hx,hz,pt)
    sys = oqupy.TimeDependentSystem(ham)

    dynamics = oqupy.compute_dynamics(system=sys,
                                      initial_state=initial_state,
                                      process_tensor=pt)
    return dynamics

if plot_dynamics:
    dynamics = get_dynamics(pt,control_parameter_results)

    t,sigma_x = dynamics.expectations(oqupy.operators.sigma('x'))
    t,sigma_y = dynamics.expectations(oqupy.operators.sigma('y'))
    t,sigma_z = dynamics.expectations(oqupy.operators.sigma('z'))

    plt.figure(2)
    plt.plot(t,sigma_x,label=r'$\sigma_x$')
    plt.plot(t,sigma_y,label=r'$\sigma_y$')
    plt.plot(t,sigma_z,label=r'$\sigma_z$')

    plt.xlabel('t [ps]')
    plt.legend()

# cost_result = cost_function(control_parameters=control_parameter_results,pt=pt)

plt.show()




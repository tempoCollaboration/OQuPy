import numpy as np
import oqupy

"""
Test dynamics with multiple identical mean-field systems are consistent.
"""

number_of_systems = 3
dt = 0.001
t_end_I = 30 * dt

# system parameters
sigma_z = oqupy.operators.sigma("z")
sigma_plus = oqupy.operators.sigma("+")
sigma_minus = oqupy.operators.sigma("-")

omega_0 = 0.0
omega_c = -30.4
Omega = 303.9

kappa = 15.2
Gamma_down = 15.2
Gamma_up = 0.8 * Gamma_down

gammas = [ lambda t: Gamma_down, lambda t: Gamma_up]
lindblad_operators = [ lambda t: sigma_minus, lambda t: sigma_plus]

def H_MF(t, a):
    return 0.5 * omega_0 * sigma_z +\
        0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)


initial_field = np.sqrt(0.05)               # Note n_0 = <a^dagger a>(0) = 0.05
initial_state = np.array([[0,0],[0,1]])     # spin down




correlations_I = oqupy.PowerLawSD(alpha=0.25,
                                  zeta=1.0,
                                  cutoff=227.9,
                                  cutoff_type="gaussian",
                                  temperature=39.3,
                                  name="ohmic")
bath_I = oqupy.Bath(0.5 * oqupy.operators.sigma("z"),
                    correlations_I,
                    name="phonon bath")

tempo_params_I = oqupy.TempoParameters(
        dt=0.001,
        tcut=0.1,
        epsrel=10**(-6))

fractions = [1/number_of_systems for i in range(number_of_systems)]
def field_eom(t, states, field):
    sx_exp_list = [np.matmul(sigma_minus, state).trace() for state in states]
    sx_exp_weighted_sum = sum([fraction*sx_exp for fraction, sx_exp in zip(fractions, sx_exp_list)])
    return -(1j*omega_c+kappa)*field - 0.5j*Omega*sx_exp_weighted_sum

system_I = oqupy.TimeDependentSystemWithField(
        H_MF,
        gammas=gammas,
        lindblad_operators=lindblad_operators)
system_list = [system_I for i in range(number_of_systems)]
mean_field_system = oqupy.MeanFieldSystem(system_list, field_eom=field_eom)

def test_mean_field_multi_sys_I():
    pt = oqupy.pt_tempo_compute(
        bath_I,
        start_time=0.0,
        end_time=t_end_I,
        parameters=tempo_params_I)
    mean_field_dynamics = oqupy.compute_dynamics_with_field(
                                    mean_field_system, 
                                    initial_field=initial_field, 
                                    initial_state_list=[initial_state for i in range(number_of_systems)], 
                                    start_time=0.0,
                                    process_tensor_list = [pt for i in range(number_of_systems)]
                                    )
    times, field_expectations = mean_field_dynamics.field_expectations()
    # check that all subsystem states are equal
    for i in range(number_of_systems - 1):
        assert np.allclose(mean_field_dynamics.system_dynamics[i].states, 
                        mean_field_dynamics.system_dynamics[i+1].states)
    
    # also check explicitly that first subsystem state is the same as last subsystem state
    # (extra check since np.allclose does not guarantee transitivity)
    assert np.allclose(mean_field_dynamics.system_dynamics[0].states, 
                        mean_field_dynamics.system_dynamics[-1].states)

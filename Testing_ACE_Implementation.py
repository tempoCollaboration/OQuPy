import oqupy
import numpy as np 
import matplotlib.pyplot as plt


sigma_x = oqupy.operators.sigma("x")
sigma_y = oqupy.operators.sigma("y")
sigma_z = oqupy.operators.sigma("z")
up_density_matrix = oqupy.operators.spin_dm("z+")
down_density_matrix = oqupy.operators.spin_dm("z-")


Omega =3.0
omega_cutoff = 3
alpha = 0.025
temperature =  5.0
s = 1.0
# beta = 1 / (0.1309 * 20)

system = oqupy.System(Omega/2 * sigma_x) 

# STEP 1: Define the continuous Spectral density.

"""
The user will have to define a spectral denisty before hand so that it can 
account for the thermalisation of the environmnets. 

This could be incorporated into already defined spectral densities but as it 
stands its currently a seperate user input

"""

def J(omega, temperature, thermalisation = False):
    
    if temperature == 0:
        J = 2.0 * alpha * omega**s * omega_cutoff**(1.0 -s) * np.exp(-omega/omega_cutoff)
        
    else:
        hbar_meV_ps = 0.6582119512777485
        kb_meV_k = 0.08617333262145
        beta = hbar_meV_ps / (temperature * kb_meV_k)
        coth = 1/np.tanh(beta * omega /2)
        if omega < 0:
            J = (-2 * alpha * np.abs(omega)**s * omega_cutoff**(1.0 -s) * np.exp(-np.abs(omega)/omega_cutoff) * (1 + coth)/2)
        else:
            J = (2 * alpha * omega**s * omega_cutoff**(1.0 -s) * np.exp(-omega/omega_cutoff) * (1 + coth)/2)
    
    return J

omega = np.linspace(0,600,10000)
j = J(omega, temperature, thermalisation = False)
plt.plot(omega, j)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$J(\omega)$')
plt.show()
# Define the spectral density paramaters


"""

These are the spectral density paramaters that will be uesd. One will need at 
a minimum OMEGA MAX which is the maximum sampled frequency in the spectral density
OMEGA MIN being the minimum sampled frequency in the spectral density typically 
set to zero. NUM MODES is the numebr of modes you wish to discretise the spectral
density by. FOCK LEVELS is the numbe of Fock levels you assign to each oscillator
this can also be given as an array so that you can define different numbr of Fock
levels for each oscillator. START TIME, the beginning time for propagation typically
set to zero. END_TIME the final you wish to propagate to. EPRSEL, the compression 
error for building the process tensor. DT the timestep.

"""

OMEGA_MIN = 0.0 # ps^{-1}
OMEGA_MAX = 30.0 #ps^{-1}
NUM_MODES = 2**5
FOCK_LEVELS = int(5.0)
START_TIME = 0.0
END_TIME = 0.2
EPSREL = 1e-7
DT = 0.0002
DRIVE = 0.0/2

"""

The following function discretises the spectral density over the defined interval 
[OMEGA_MIN, OMEGA_MAX] for a give number of modes NUM_MODES. The strategy for 
discretisation can also be changed from the standard Gauss-Legendre discretisation.
The three methods that can be done is equidistant, gauss_legendre and chain_mapping. 
Chain mapping is the most recent advanced and is adapted from the TEDOPA style for
discretising a continous environments. If you want the spectral density to be thermalised
so that you can intialise the modes in the ground state instead of the thermal Gibbs state
then you can turn this on, bare in mind this will double the sampling window to account for 
negative frequency modes and therefore may require a larger number of modes to accurately 
discretise the spectral density. The function returns the frequenices of the oscillators
and the coupling strength of the oscillators to the system

"""


frequencies, couplings = oqupy.discretize_spectral_density(
    spectral_density=J,
    omega_min=0.0,
    omega_max=OMEGA_MAX,
    num_modes=NUM_MODES,
    temperature=temperature,
    thermalisation = True,
    discretisation_strategy="chain_mapping",
)


plt.figure(figsize = (6,4), dpi = 100)
plt.scatter(frequencies, couplings)
plt.plot(frequencies, np.abs(couplings))
plt.xlabel(r'$\omega$ ps$^{-1}$')
plt.ylabel(r'$J(\omega)$ ps$^{-1}$')
plt.show()

"""

The following functons plot the bath correlation function with the discretised modes 
that were found from above. This can be useful to determine whether you have accuretly 
discretised the spectral density and have repreoduced the bath correlation function you would 
expec to get 
"""

bath_correlation_function = oqupy.discrete_bath_correlation_function(frequencies, couplings, START_TIME, END_TIME, DT)

fig = oqupy.plot_bath_correlation_function(START_TIME, END_TIME, DT, bath_correlation_function)


"""

This function constructs the modes that are going to be combied and compressed into 
the Process Tensor

"""

modes = [oqupy.AceMode.harmonic(
    omega=frequency, coupling_strength=coupling,
    n_levels=FOCK_LEVELS, temperature=temperature,thermalisation = True)
    for frequency, coupling in zip(frequencies, couplings)]



"""

This function then calculates the Process Tensor using the modes defined previously 
sequential is the standard combination of modes described in the original ACE paper
Nature Physics volume 18, pages662-668 (2022). tree_like compression combines pairs
of modes together as in Phys. Rev. Research 6, 043203. Once the combination method
has been choosen then the compression method can be choosen with either the standard
forward and backward sweeep or through preselection described in  Phys. Rev. X 14, 011010


"""

process = oqupy.ace_compute(
    coupling_operator=up_density_matrix, modes=modes,
    start_time=0.0, end_time=END_TIME,
    parameters=oqupy.AceParameters(DT, EPSREL, method_combination="tree", method_svd= 'preselection'),
    progress_type="bar")



"""

From here onwards its stanadrd OQuPy implementation to caluclat the dynamics 
etc

"""

sigma_x = oqupy.operators.sigma("x")
sigma_z = oqupy.operators.sigma("z")
initial_state = np.diag([0.0, 1.0]) #* 1/np.sqrt(2)
# initial_state = 0.5 * np.array([
#     [0.0, 0.0],
#     [1.0, 0.0]
# ], dtype=complex)

DRIVE = 3.0/2
dynamics = oqupy.compute_dynamics(
    system=oqupy.System(DRIVE * sigma_x), initial_state=initial_state,
    process_tensor=process, progress_type="bar")


down_density_matrix = oqupy.operators.spin_dm("z-")
up_density_matrix = oqupy.operators.spin_dm('z+')

# density_matrix = 1/2 * (up_density_matrix + down_density_matrix)


expectation = np.array([[1,0],[0,0]])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size': 18
})

t, s_z = dynamics.expectations(expectation, real=False)
times, states = dynamics.times, dynamics.states

rho_10 = states[:,1,0]


if not np.all(np.isfinite(states)):
    raise FloatingPointError("Dynamics contains nonfinite values.")
trace_error = np.max(np.abs(np.trace(states, axis1=1, axis2=2) - 1))
print(f"Maximum trace error: {trace_error:.3e}")
print("Maximum MPO bond dimension:", max(process.get_bond_dimensions()))


fig, ax = plt.subplots(figsize=(6, 6), dpi = 100)
ax.plot(t, (s_z), linewidth=2, label = 'ACE OQuPy', color = '#D81B60')
ax.set_xlabel("Time (ps)")
ax.set_ylabel(r"$\langle 1 | \rho(t) | 1 \rangle $")
plt.legend(frameon = False)
plt.show()

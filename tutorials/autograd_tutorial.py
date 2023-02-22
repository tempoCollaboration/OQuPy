
import numpy as np

import oqupy

from oqupy.process_tensor import BaseProcessTensor

from scipy.linalg import expm
from scipy.interpolate import interp1d

from jax.scipy.linalg import expm as jexpm
# from jax_routines import commutator
from jax import grad, jacfwd
from jax import numpy as jnp
from oqupy.helpers import get_half_timesteps, get_full_timesteps


from typing import Callable

process_tensor = oqupy.import_process_tensor(
            'optimisation_pt.processTensor','simple')

def jcommutator(operator: np.ndarray) -> np.ndarray:
    """Construct commutator superoperator from operator.
    however using jax operators rather than numpy ones"""
    dim = operator.shape[0]
    return jnp.kron(operator, jnp.identity(dim)) \
            - jnp.kron(jnp.identity(dim), jnp.transpose(operator))



def dpropagator(hamiltonian: Callable[[float],np.ndarray],
    t: float, # half propagator time
    dt: float,
    op: np.ndarray, # operator to differentiate wrt
    h: float): # finite difference time separation
    '''
    deriv of propagator wrt the pre node and the post node
    '''

    liouvillian_plus_h=-1j * oqupy.operators.commutator(hamiltonian(t)+h*op)
    liouvillian_minus_h=-1j * oqupy.operators.commutator(hamiltonian(t)-h*op)

    propagator_plus_h=expm(liouvillian_plus_h*dt/2.0).T
    propagator_minus_h=expm(liouvillian_minus_h*dt/2.0).T

    deriv=(propagator_plus_h-propagator_minus_h)/(2.0*h)

    return deriv

def jax_dpropagator(hamiltonian: Callable[[float],np.ndarray],
    t: float, # half propagator time
    dt: float,
    op: np.ndarray): # operator to differentiate wrt

    def liouvillian_at_t(operator):
        res = jnp.transpose(
            jexpm(-1j * jcommutator(hamiltonian(t) + operator) * dt/2))
        return res



    # liouvillian_at_t = lambda operator: -1j * jcommutator(hamiltonian)
    gradient = jacfwd(liouvillian_at_t,holomorphic=True)

    actual_gradient = gradient(op)

    sum1 = np.sum(actual_gradient,axis=-1)
    sum2 = np.sum(sum1,axis=-1)


    return sum2.T


def get_hamiltonian(hx:np.ndarray,hz:np.ndarray,pt:BaseProcessTensor):

    expval_times = get_full_timesteps(pt,0)
    assert hx.size == expval_times.size, 'hx must be same length as number of timesteps'
    assert hz.size == expval_times.size, 'hz must be same length as number of timesteps'
    # interp doesn't extrapolate beyond the last data point so any time after t-dt will
    # be out of bounds, need to duplicate the final point so we create the very last 'pixel'
    # expval_times plus one timestep
    expval_times_p1 = np.concatenate((expval_times,np.array([pt.dt * len(pt)])))
    # duplicate last element so any time between t_f-dt and t_f falls within this 'pixel'
    # otherwise scipy interp1d doesn't like extrapolating so calls it out of bounds
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

times = get_full_timesteps(process_tensor,start_time=0)
max_time = 5
# pi pulse conjugate to s_z
h_x = np.ones(times.size) *np.pi / max_time
h_z = np.zeros(times.size)
hamiltonian_t = get_hamiltonian(hx=h_x,hz=h_z,pt=process_tensor)

t = 1.0 # above pulse is time independent so this doesn't matter

matrix = np.array([[0.2,0.5],[0.5,0.3]],dtype='complex128')

fd_deriv = dpropagator(hamiltonian=hamiltonian_t,
                       t=t,
                       dt=process_tensor.dt,
                       op=matrix,
                       h=10^(-6))

np.set_printoptions(precision=3,suppress=True)

print(fd_deriv.imag)

jax_deriv = jax_dpropagator(hamiltonian=hamiltonian_t,
                       t=t,
                       dt=process_tensor.dt,
                       op=matrix)

print(jax_deriv.imag)

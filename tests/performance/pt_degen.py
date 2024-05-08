#!/usr/bin/env python

def pt_degen_performance_A(spin_size, unique):
    correlations = oqupy.PowerLawSD(alpha=p['alpha'],
                                      zeta=p['zeta'],
                                      cutoff=p['cutoff'],
                                      cutoff_type=cutoff_type,
                                      temperature=p['T'])
    bath = oqupy.Bath(oqupy.operators.sigma("z")/2.0, correlations)
    pt_tempo_parameters = oqupy.TempoParameters(dt=dt, tcut=p['tcut'], epsrel=epsrel)
    t0 = time()
    pt_tempo_compute(unqiue=unique)
    result['walltime'] = time()-t0
    result['unique'] = unique
    result['spin_size'] = spin_size
    pt.close()
    return result


parameters_A1 = [[2, 4, 8], [True, False]]

ALL_TESTS = [ (pt_degen_performance_A, [parameters_A1]),
             ]


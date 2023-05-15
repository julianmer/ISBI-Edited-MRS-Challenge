####################################################################################################
#                                          simulation.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 15/12/22                                                                                #
#                                                                                                  #
# Purpose: Simulate a corpus of MRS data.                                                          #
#                                                                                                  #
####################################################################################################



#*************#
#   imports   #
#*************#
import numpy as np

# own
from simulation.simulationDefs import cleanParams, stdConcs


#***********************************************#
#   eliminate randomness to reproduce results   #
#***********************************************#
np.random.seed(42)


#*********************#
#   draw parameters   #
#*********************#
def simulateParam(basis, batch, params=cleanParams, concs=stdConcs):
    """
    Function to simulate SVS parameters.

    @param basis -- The basis set of metabolites to simulate as (FSL) MRS object.
    @param batch -- The number of samples.
    @param params -- The simulation parameters in form of a dictionary,
                     if not given internal parameter configuration will be used.
    @param concs -- The concentration ranges of the metabolites in form of a dictionary,
                    if not given standard concentrations will be used.

    @returns -- The parameters.
    """
    if params['dist'] == 'unif':
        dist = np.random.uniform
    elif params['dist'] == 'normal':
        dist = np.random.normal

    # get metabolite concentrations
    randomConc = {}
    for name in basis.names:
        cName = name.split('.')[0]   # remove format ending (e.g. 'Ace.raw' -> 'Ace')

        #  draw randomly from range
        randomConc[name] = dist(concs[cName]['low_limit'], concs[cName]['up_limit'], batch)

    gamma = dist(params['broadening'][0][0], params['broadening'][1][0], batch)
    sigma = dist(params['broadening'][0][1], params['broadening'][1][1], batch)
    shifting = dist(params['shifting'][0], params['shifting'][1], batch)
    phi0 = dist(params['phi0'][0], params['phi0'][1], batch)
    phi1 = dist(params['phi1'][0], params['phi1'][1], batch)

    theta = np.array(list(randomConc.values()))
    theta = np.concatenate((theta, gamma[np.newaxis, :]))
    theta = np.concatenate((theta, sigma[np.newaxis, :]))
    theta = np.concatenate((theta, shifting[np.newaxis, :]))
    theta = np.concatenate((theta, phi0[np.newaxis, :]))
    theta = np.concatenate((theta, phi1[np.newaxis, :]))

    for i in range(len(params['baseline'][0])):
        theta = np.concatenate((theta, dist(params['baseline'][0][i],
                                            params['baseline'][1][i], batch)[np.newaxis, :]))


    noiseCov = [[10 * np.random.normal(params['noiseCov'][0], params['noiseCov'][1])]]


    noise = np.random.multivariate_normal(np.zeros((1)), noiseCov,
                                          (batch, basis.n)) \
            + 1j * np.random.multivariate_normal(np.zeros((1)), noiseCov,
                                                 (batch, basis.n))

    noise2 = np.random.multivariate_normal(np.zeros((1)), noiseCov,
                                          (batch, basis.n)) \
            + 1j * np.random.multivariate_normal(np.zeros((1)), noiseCov,
                                                 (batch, basis.n))

    return np.swapaxes(theta, 0, 1), noise[:, :, 0], noise2[:, :, 0]


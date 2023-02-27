##========================== PATH DEPENDENT DEPTH 3 PARALLELIZED: MODIFIED =========================##
import time
import numexpr as ne
import numpy as np
from numba import njit, int64, float64, prange
from numba.experimental import jitclass

# -----------------------------------------------------------------------------------------------------------------------
# parameters

spec = {
    'd': int64,
    'r': float64,
    'T': int64,
    'sigma': float64,
    'tau': float64,
    'kappa': float64,
    'rho': float64,
    'B': float64,
    'x0': float64,
    'num_path': int64,
    'num_path_2': int64,
    'num_path_3': int64,
    'num_path_4': int64,
    'num_path_5': int64,
    'threshold': float64,
    'discount_vet': float64[:],
}


@jitclass(spec)
class Parameters:
    """
    the class included all parameters of the Bermudan max call instance
    """

    def __init__(self):
        self.d = 4
        self.r = 0.05
        self.T = 55
        self.sigma = 0.2
        self.tau = 1 / 18
        self.kappa = 100
        self.rho = -0.05
        self.B = 170
        self.x0 = 100
        self.discount_vet = np.exp(- self.r * self.tau * np.arange(self.T))
        self.num_path = 10 ** 5 * 3
        self.num_path_2 = 10 ** 5
        self.num_path_3 = 10 ** 2
        self.num_path_4 = 10 ** 2
        self.num_path_5 = 10 ** 2
        self.threshold = 0.08


P = Parameters()


# -----------------------------------------------------------------------------------------------------------------------
# utlilty functions

@njit
def row_max_2d(C):
    return np.array([np.max(C[i, :]) for i in range(C.shape[0])])


@njit
def row_min_2d(C):
    return np.array([np.min(C[i, :]) for i in range(C.shape[0])])


@njit
def row_max_3d(C):
    a = np.zeros((C.shape[0], C.shape[2]))
    for i in range(C.shape[0]):
        for j in range(C.shape[2]):
            a[i, j] = np.max(C[i, :, j])
    return a


@njit
def row_cumprod_2d(C):
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] == 0:
                C[i, j:] = 0
                break
    return C


#
# @njit
# def positive_part_2d(C):
#     return np.array([[np.max(C[i, j], 0) for i in range(C.shape[0])] for j in range(C.shape[1])])


# -----------------------------------------------------------------------------------------------------------------------
# simulator and payoff

@njit
def payoff(gamma, P, discount_vet=P.discount_vet):
    """

    :param P:
    :param discount_vet:
    :param gamma: an m by d by T matrix which is m sample paths from the GBM,
    :return: an m by T array: the (i, t)-th element being the payoff if stopping at time t on sample path i
    """

    # a = np.zeros((gamma.shape[0], gamma.shape[2]))
    # for j in range(gamma.shape[2]):
    #     a[:, j] = row_max_2d(gamma[:, :, j]) - P.kappa
    # return np.maximum(a, 0) * discount_vet
    a = row_max_3d(gamma)
    return np.maximum(a - P.kappa, 0) * row_cumprod_2d(a < P.B) * discount_vet

    # the t-th payoff is given by  e^{- r t} * S_t/ S_{t - 100}


class Simulator:
    """

    simulate num_path * (T + 99) independent standard normal rvs
    also compute a by num_path by (T + 100) cumulative sample paths array
    """

    def __init__(self):
        dW = np.random.normal((P.r - P.sigma ** 2 / 2) * P.tau,
                              P.sigma * np.sqrt(P.tau), (P.num_path, P.d, P.T - 1))
        # a num_path by d by T-1 normally distributed array with the specified (by GBM) mean and variance
        correlationmatrix = np.ones((P.d, P.d)) * P.rho + (1 - P.rho) * np.identity(P.d)

        dq = np.zeros((P.num_path, P.d, P.T))
        dq[:, :, 1:] = np.cumsum(np.matmul(correlationmatrix, dW), axis=2)
        dq = P.x0 * np.exp(dq)
        # for each i, dQ[i] gives an independent cumulative dynamics of the gbm
        self.normal_matrix = dW
        self.cumulative_path = dq


# -----------------------------------------------------------------------------------------------------------------------
# simulating sample paths

w = Simulator()
dQ = w.cumulative_path


# -----------------------------------------------------------------------------------------------------------------------
# depth one

@njit
def depth_1(gamma, P, start=0):
    """
    :param gamma: a m by P.T  array, gives the cumulative paths from the Simulator
    :param P: parameters
    :param start: an integer within [0, P.T), only compute depth1 values for times from "start"
    :return: depth1 value of the input sample paths starting from "start", computed directly from the payoff function
    """
    return payoff(gamma[:, :, start:], P, P.discount_vet[start:])


# -----------------------------------------------------------------------------------------------------------------------
# depth two

@njit
def depth_2_i(gamma_i, depth1_i, P, start=0):
    """
    :param start: an integer within [0, P.T), only compute depth1 values for times from "start"
    :param gamma_i:a P.d by P.T  array, gives the cumulative paths from the Simulator
    :param depth1_i: a (P.T) array, gives the depth1 values of gamma_i path
    :param P: parameters
    :return: depth2 value of the input sample path starting from "start"
    """

    depth2_i = np.copy(depth1_i)

    #     if np.min(depth1_i) > 0.001:
    sample_matrix = np.empty((P.num_path_3, P.d, P.T))
    sample_matrix[:] = gamma_i

    for t in range(P.T - 2, start - 1, -1):
        rand_starter = np.random.randint(0, P.num_path - P.num_path_3 + 1)
        sample_matrix[:, :, t + 1:] = (dQ[rand_starter: rand_starter + P.num_path_3, :, t + 1:].T
                                       / (dQ[rand_starter: rand_starter + P.num_path_3, :, t] /
                                          sample_matrix[0, :, t]).T).T

        b = depth_1(sample_matrix, P, start=t + 1)  # avoid duplicate computation
        depth2_i[t] = np.mean(np.maximum(np.max(depth1_i[:t + 1]), row_max_2d(b))) - depth1_i[t]
    depth2_i[P.T - 1] = np.max(depth1_i) - depth1_i[P.T - 1]
    return depth2_i[start:]


@njit(parallel=True)
def depth_2(gamma, P, start=0):
    """
    :param gamma: an m by P.d by P.T array, gives the cumulative paths from the Simulator
    :param P: parameters
    :param start: an integer within [0, P.T), only compute depth2 values for times from "start"
    :return: depth1 value in full, and depth2 value of the input sample paths starting from "start"
    """

    depth1 = depth_1(gamma, P)

    num_path_2 = gamma.shape[0]
    depth2 = np.zeros((num_path_2, P.T))

    for i in prange(num_path_2):
        depth2[i, start:] = depth_2_i(gamma[i], depth1[i], P, start)

    return depth1, depth2[:, start:]


@njit
def faster_depth_2_i(gamma_i, depth1_i, P, start=0):
    """
    :param gamma_i: a P.d by P.T array, gives one sample path from the Simulator
    :param depth1_i: a (P.T) array, gives the depth1 values of gamma_i path
    :param P: parameters
    :param start: an integer within [0, P.T), only compute depth2 values for times from "start"
    :return: depth2 value of the input sample path starting from "start"
    """

    #     depth2_i = np.copy(depth1_i)
    #     if np.min(depth1_i) > 0.001:
    sample_matrix = np.empty((P.num_path_3, P.d, P.T))
    depth2_i = np.zeros(P.T)
    sample_matrix[:] = gamma_i

    for t in range(P.T - 2, - 1, -1):
        rand_starter = np.random.randint(0, P.num_path - P.num_path_3 + 1)
        sample_matrix[:, :, t + 1:] = (dQ[rand_starter: rand_starter + P.num_path_3, :, t + 1:].T
                                       / (dQ[rand_starter: rand_starter + P.num_path_3, :, t] /
                                          sample_matrix[0, :, t]).T).T

        b = row_max_2d(depth_1(sample_matrix, P, start=t + 1))
        depth2_i[t:] += np.mean(np.maximum(b, depth1_i[t]))
        depth2_i[t + 1:] -= np.mean(b)
    depth2_i[start:P.T - 1] -= depth1_i[start:P.T - 1]

    return depth2_i[start:]


@njit(parallel=True)
def faster_depth_2(gamma, P, start=0):
    """
    :param gamma: a (m, P.d, P.T) array, gives the cumulative paths from the Simulator
    :param P: parameters
    :param start: an integer within [0, P.T), only compute depth2 values for times from "start"
    :return: depth1 value in full, and depth2 value of the input sample paths starting from "start"
    """

    depth1 = depth_1(gamma, P)

    num_path_2 = gamma.shape[0]
    depth2 = np.zeros((num_path_2, P.T))

    for i in prange(num_path_2):
        depth2[i, start:] = faster_depth_2_i(gamma[i], depth1[i], P, start)

    return depth1, depth2[:, start:]


# -----------------------------------------------------------------------------------------------------------------------
# depth three

@njit
def depth_3_i(gamma_i, depth2_i, P, start=0):
    """
    :param gamma_i: a (P.d, P.T) array, gives one sample path from the Simulator
    :param depth2_i: a (P.T) array, gives the depth2 values of gamma_i path
    :param P: parameters
    :param start: an integer within [0, P.T), only compute depth3 values for times from "start"
    :return: depth3 value of the input sample path starting from "start"
    """

    depth3_i = np.copy(depth2_i)

    if np.min(depth2_i) > 0.00001:
        sample_matrix = np.empty((P.num_path_5, P.d, P.T))
        sample_matrix[:] = gamma_i

        for t in range(P.T - 2, start - 1, -1):
            rand_starter = np.random.randint(0, P.num_path - P.num_path_5 + 1)
            sample_matrix[:, :, t + 1:] = (dQ[rand_starter: rand_starter + P.num_path_5, :, t + 1:].T
                                           / (dQ[rand_starter: rand_starter + P.num_path_5, :, t] /
                                              sample_matrix[0, :, t]).T).T

            dummy, b = depth_2(sample_matrix, P, start=t + 1)  # avoid duplicate computation
            depth3_i[t] = depth2_i[t] - np.mean(np.minimum(np.min(depth2_i[:t + 1]), row_min_2d(b)))
        depth3_i[P.T - 1] = depth2_i[P.T - 1] - np.min(depth2_i)

    return depth3_i[start:]


@njit(parallel=True)
def depth_3(gamma, P, start=0):
    """
    :param gamma: a (m, P.d, P.T) array, gives the cumulative paths from the Simulator
    :param P: parameters
    :param start: an integer within [0, P.T), only compute depth3 values for times from "start"
    :return: depth1 and depth2 value in full, and depth3 value of the input sample paths starting from "start"
    """

    depth1, depth2 = depth_2(gamma, P)

    num_path_4 = gamma.shape[0]
    depth3 = np.zeros((num_path_4, P.T))

    for i in prange(num_path_4):
        depth3[i, start:] = depth_3_i(gamma[i], depth2[i], P, start)
    return depth1, depth2, depth3[:, start:]


@njit
def faster_depth_3_i(gamma_i, depth2_i, P, start=0):
    """
    :param gamma_i: a (P.d, p.T) array, gives one sample path from the Simulator
    :param depth2_i: a (P.T) array, gives the depth2 values of gamma_i path
    :param P: parameters
    :param start: an integer within [0, P.T), only compute depth3 values for times from "start"
    :return: depth3 value of the input sample path starting from "start"
    """

    depth3_i = np.copy(depth2_i)
    if np.min(depth2_i) > 0.00001:
        sample_matrix = np.empty((P.num_path_5, P.d, P.T))
        depth3_i[:] = 0.
        sample_matrix[:] = gamma_i

        for t in range(P.T - 2, - 1, -1):
            rand_starter = np.random.randint(0, P.num_path - P.num_path_5 + 1)
            sample_matrix[:, :, t + 1:] = (dQ[rand_starter: rand_starter + P.num_path_5, :, t + 1:].T
                                           / (dQ[rand_starter: rand_starter + P.num_path_5, :, t] /
                                              sample_matrix[0, :, t]).T).T

            dummy, depth2_t = depth_2(sample_matrix, P, start=t + 1)
            b = row_min_2d(depth2_t)
            depth3_i[t:] -= np.mean(np.minimum(b, depth2_i[t]))
            depth3_i[t + 1:] += np.mean(b)
        depth3_i[start:P.T - 1] += depth2_i[start:P.T - 1]

    return depth3_i[start:]


@njit(parallel=True)
def faster_depth_3(gamma, P, start=0):
    """
    :param gamma: a (m, P.d, P.T) array, gives the cumulative paths from the Simulator
    :param P: parameters
    :param start: an integer within [0, P.T), only compute depth3 values for times from "start"
    :return: depth1 and depth2 value in full, and depth3 value of the input sample paths starting from "start"
    """

    depth1, depth2 = faster_depth_2(gamma, P)

    num_path_4 = gamma.shape[0]
    depth3 = np.zeros((num_path_4, P.T))

    for i in prange(num_path_4):
        depth3[i, start:] = faster_depth_3_i(gamma[i], depth2[i], P, start)
    return depth1, depth2, depth3[:, start:]


# -----------------------------------------------------------------------------------------------------------------------
# main functions

def main_cal_2():
    t2 = time.time()
    r1 = 0 if P.num_path == P.num_path_2 else np.random.randint(0, P.num_path - P.num_path_2)

    t0 = time.time()
    depth1 = depth_1(dQ, P)
    print("time to depth1 ", time.time() - t0)

    t1 = time.time()
    dummy, depth2 = depth_2(dQ[r1:r1 + P.num_path_2], P)
    print("time to parallel depth2 ", time.time() - t1)

    # t3 = time.time()
    # dummy, depth2_faster = faster_depth_2(dQ[r1:r1 + P.num_path_2], P)
    # print("time to parallel depth2_faster ", time.time() - t3)

    np.save('kmc_depth2_dummy1', dummy)
    np.save('kmc_depth2', depth2)
    # np.save('kmc_depth2_faster', depth2_faster)

    # ----------------------------------------------------------------------------------------------------------------------
    # outputs
    e1 = np.mean(np.max(depth1, axis=1))
    std1 = np.std(np.max(depth1, axis=1))
    e2 = e1 - np.mean(np.min(depth2, axis=1))
    std2 = np.std(np.min(depth2, axis=1))

    # e2_faster = e1 - np.mean(np.min(depth2_faster, axis=1))
    # std2_faster = np.std(np.min(depth2_faster, axis=1))

    print("total runtime", time.time() - t2)
    print("depth one mean =", e1)
    print("depth one std =", std1)
    print("depth two mean =", e2)
    print("depth two std =", std2)
    print()
    # print("faster expansion depth two mean =", e2_faster)
    # print("faster expansion depth two std =", std2_faster)


def main_cal_3():
    t2 = time.time()
    r1 = 0 if P.num_path == P.num_path_2 else np.random.randint(0, P.num_path - P.num_path_2)
    r2 = 0 if P.num_path == P.num_path_4 else np.random.randint(0, P.num_path - P.num_path_4)

    t0 = time.time()
    depth1 = depth_1(dQ, P)
    print("time to depth1 ", time.time() - t0)

    t4 = time.time()
    dummy1, dummy2, depth3 = depth_3(dQ[r2:r2 + P.num_path_4], P)
    print("time to parallel depth3 ", time.time() - t4)

    t5 = time.time()
    dummy1, dummy2, depth3_faster = faster_depth_3(dQ[r2:r2 + P.num_path_4], P)
    print("time to parallel depth3_faster ", time.time() - t5)

    t1 = time.time()
    dummy, depth2 = depth_2(dQ[r1:r1 + P.num_path_2], P)
    print("time to parallel depth2 ", time.time() - t1)

    t3 = time.time()
    dummy, depth2_faster = faster_depth_2(dQ[r1:r1 + P.num_path_2], P)
    print("time to parallel depth2_faster ", time.time() - t3)

    np.save('knock_out_max_call_depth3', depth3)
    np.save('knock_out_max_call_depth3_faster', depth3_faster)

    # ----------------------------------------------------------------------------------------------------------------------
    # outputs
    e1 = np.mean(np.max(depth1, axis=1))
    std1 = np.std(np.max(depth1, axis=1))
    e2 = e1 - np.mean(np.min(depth2, axis=1))
    std2 = np.std(np.min(depth2, axis=1))
    e3 = e2 - np.mean(np.min(depth3, axis=1))
    std3 = np.std(np.min(depth3, axis=1))

    e2_faster = e1 - np.mean(np.min(depth2_faster, axis=1))
    std2_faster = np.std(np.min(depth2_faster, axis=1))
    e3_faster = e2_faster - np.mean(np.min(depth3_faster, axis=1))
    std3_faster = np.std(np.min(depth3_faster, axis=1))

    print("total runtime", time.time() - t2)
    print("depth one mean =", e1)
    print("depth one std =", std1)
    print("depth two mean =", e2)
    print("depth two std =", std2)
    print("depth three mean =", e3)
    print("depth three std =", std3)
    print()
    print("faster expansion depth two mean =", e2_faster)
    print("faster expansion depth two std =", std2_faster)
    print("faster expansion depth three mean =", e3_faster)
    print("faster expansion depth three std =", std3_faster)


def policy(depth1, depthk, p, q, r):
    """

    :param depth1: the first output from depth_k
    :param depthk: the second output from depth_k
    :return: the performance of a threshold policy: the mean value, the standard deviation
    """

    length_2 = len(depthk.T[0])
    length_3 = len(depthk[0])
    policy_performance = np.zeros(length_2)
    policy_marker = np.zeros(length_2)

    for t in range(length_3):
        for i in range(length_2):
            if (policy_marker[i] == 0) and (
                    (depth1[i, t] - p * depthk[i, t] + q * t > r) or t == length_3 - 1):
                policy_marker[i] = 5
                policy_performance[i] = depth1[i, t]

    u = np.mean(policy_performance)
    v = np.std(policy_performance)

    return u, v


v = np.zeros(10)
for t in range(10):
    main_cal_2()
    a = np.load('kmc_depth2_dummy1.npy')
    b = np.load('kmc_depth2.npy')
    u, _ = policy(a, b, 2.74, 0.03, 32.3)
    v[t] = u
    print(u)
    print()

print(np.mean(v), np.std(v))

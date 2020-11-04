"""
Created on  Oct 21 2020
@author: wangmeng
"""
# -------------------------- Simulation Environment ------------------------------#
from pylab import *
import cvxpy as cvx
import math
import numpy as np
import random
import warnings
np.random.seed(1)
random.seed(1)
def phase_from_subset(M,N):
    rd = np.random.RandomState(1)
    phase_H = np.exp(1j * 2 * math.pi * rd.random(size=(M, N)))
    phase_g = np.exp(1j * 2 * math.pi * rd.random(size=(M,1)))
    phase_f = np.exp(1j * 2 * math.pi * rd.random(size=(N,1)))

    phase_g1 = np.exp(1j * 2 * math.pi * rd.random(size=(M, 1)))
    phase_f1 = np.exp(1j * 2 * math.pi * rd.random(size=(N, 1)))

    phase_g2 = np.exp(1j * 2 * math.pi * rd.random(size=(M, 1)))
    phase_f2 = np.exp(1j * 2 * math.pi * rd.random(size=(N, 1)))
    return phase_g, phase_H, phase_f , phase_g1, phase_f1, phase_g2, phase_f2

def LogdistancePropagation(d, d0, L0, n):
    if d<d0:
        L = L0 * np.ones(np.size(d))
    else:
        L = L0 + 10 * n * math.log(d/ d0 , 10)
    g = math.sqrt(10. ** (-L / 10))
    return g


def environment(state, action, config):
    state_next = state  + action
    done = 0
    theta_all = []
    snr_all = []

    # ---------------channel_state---------------#
    M = config.M
    N = config.N
    # 1.channels
    AP = config.AP
    USER0 = config.USER0
    USER1 = config.USER1
    USER2 = config.USER2
    IRS = state_next

    phase_g, phase_H, phase_f, phase_g1, phase_f1, phase_g2, phase_f2 = phase_from_subset(M, N)

    d_AI0 = math.sqrt((AP[0] - IRS[0]) ** 2 + (AP[1] - IRS[1]) ** 2 + (AP[2] - IRS[2]) ** 2)
    d_IU0 = math.sqrt((IRS[0] - USER0[0]) ** 2 + (IRS[1] - USER0[1]) ** 2 + (IRS[2] - USER0[2]) ** 2)
    d_AU0 = math.sqrt((AP[0] - USER0[0]) ** 2 + (AP[1] - USER0[1]) ** 2 + (AP[2] - USER0[2]) ** 2)

    d_IU1 = math.sqrt((IRS[0] - USER1[0]) ** 2 + (IRS[1] - USER1[1]) ** 2 + (IRS[2] - USER1[2]) ** 2)
    d_AU1 = math.sqrt((AP[0] - USER1[0]) ** 2 + (AP[1] - USER1[1]) ** 2 + (AP[2] - USER1[2]) ** 2)

    d_IU2 = math.sqrt((IRS[0] - USER2[0]) ** 2 + (IRS[1] - USER2[1]) ** 2 + (IRS[2] - USER2[2]) ** 2)
    d_AU2 = math.sqrt((AP[0] - USER2[0]) ** 2 + (AP[1] - USER2[1]) ** 2 + (AP[2] - USER2[2]) ** 2)

    # 2. channel propogation, log distance model
    d_ref = config.d_ref  # reference distance
    exponent_AU = config.exponent_AU  # degradation exponent
    exponent_AI = config.exponent_AI  # degradation exponent
    exponent_IU = config.exponent_IU  # degradation exponent
    L0 = config.L0  # pass loss at the reference distance
    gain = config.gain
    Ld = config.Ld  # extra loss at direct link

    amplify_H = LogdistancePropagation(d_AI0, d_ref, L0 - gain, exponent_AI)
    amplify_g0 = LogdistancePropagation(d_AU0, d_ref, L0 + Ld, exponent_AU)
    amplify_f0 = LogdistancePropagation(d_IU0, d_ref, L0, exponent_IU)

    amplify_g1 = LogdistancePropagation(d_AU1, d_ref, L0 + Ld, exponent_AU)
    amplify_f1 = LogdistancePropagation(d_IU1, d_ref, L0, exponent_IU)

    amplify_g2 = LogdistancePropagation(d_AU2, d_ref, L0 + Ld, exponent_AU)
    amplify_f2 = LogdistancePropagation(d_IU2, d_ref, L0, exponent_IU)

    H = amplify_H * phase_H
    g0 = amplify_g0 * phase_g
    f0 = amplify_f0 * phase_f

    g1 = amplify_g1 * phase_g1
    f1 = amplify_f1 * phase_f1

    g2 = amplify_g2 * phase_g2
    f2 = amplify_f2 * phase_f2

    rho = config.rho

    # ---------------多用户遍历---------------#

    for i in range(3):
        if i ==0 :
            H_f = H @ np.diag(f0.reshape(-1))
            H_f_mat = mat(H_f)
            g_mat = mat(g0)
        elif i == 1:
            H_f = H @ np.diag(f1.reshape(-1))
            H_f_mat = mat(H_f)
            g_mat = mat(g1)
        elif i==2:
            H_f = H @ np.diag(f2.reshape(-1))
            H_f_mat = mat(H_f)
            g_mat = mat(g2)

        # ---------------CVX求theta---------------#
        phi = rho * conjugate(H_f_mat).T  # 20*2
        b = phi * g_mat
        c = conjugate(g_mat).T * conjugate(phi).T  # 注意这里是共轭转置
        d = mat(0)
        a = phi * conjugate(phi).T  # 20*20


        R1 = hstack((a, b))  # 按行合并，即行数不变，扩展列数
        R2 = hstack((c, d))  # 按行合并，即行数不变，扩展列数
        R = vstack((R1, R2))  # 按列合并，即增加行数 21 * 21

        X = cvx.Variable((N+1, N+1), hermitian=True)
        constr = [X >> 0]
        for t in range(N + 1):
            constr += [X[t, t] - 1 == 0]

        prob = cvx.Problem(cvx.Maximize(cvx.real(cvx.trace(R @ X))), constr)

        prob.solve()

        # decompose
        try:
            eigval, eigvec = np.linalg.eig(X.value) #极个别情况会出现prob无解的情况，为了防止在迭代意外终止加此项
        except:
            state_next = state #此时reward不计，返回上一状态
            return 0, state_next, 0 ,1

        pos = argmax(abs(eigval))  # 这里的eigval的顺序和matlab中是相反的

        # 取特征值最大的那一列特征向量
        theta = eigvec[:, pos] / norm(eigvec[:, pos]) * sqrt(real(trace(X.value)))#这里计算之后会把特征向量写成行向量的行式

        theta = theta[0:len(theta)-1]#将最后一位扔掉 extended theta with last element as 1

        theta_all.append(theta)
        print(np.size(theta_all))

        theta_mat = mat(theta).T

        # ---------------Action---------------#

        w = np.array([[complex(10, -5)], [complex(10, -5)]])# 3.23685584 - 2.47857043 4.12022957 0.01185447
        w_mat = mat(w)

        # -------------AUV_station - -----------#
        area_flag = 0
        if (state_next[0] < 0) | (state_next[0] > 20):
            area_flag = 1
        if (state_next[1] < -10) | (state_next[1] > 10):
            area_flag = 1
        if (state_next[2] < 0) | (state_next[2] > 20):
            area_flag = 1
        if area_flag:
            return 0, state, done, theta
        #-------------snr_min - -----------#
        snr_min_ = (np.linalg.norm(np.conj(g_mat + rho * H_f_mat * theta_mat).T * w_mat, ord=2))#可以直接在matlab中计算，看看是否比这种方式快
        snr_min = snr_min_ * snr_min_
        snr_all.append(snr_min)

    i = argmax(snr_all)
    print(i)

    snr = snr_all[i]


    #-------------energy_min - -----------#
    # energy_min_ = (np.linalg.norm(np.conj(H_mat).T * w_mat, ord=2))
    # energy_min = config.eta * (1 - rho * rho) * energy_min_ * energy_min_

    # ---------------Reward---------------
    # if snr_min < 1e-7:  # SNR约束
    #     print('SNR not OK')
    #     # next_state = channel_state(action[0], action[1], action[2])
    #     reward = -3500  # 如果不符合约束，则输出一个绝对值很大的负值。该负值需要小于算法所获得的最小值，基于实验结果进行调试。
    #     state_next = [5, 0, 0]  # 不符合约束，结束探索，返回起点
    #     done = 1
    #     return reward, state_next, done
    # else:
    #     # print('SNR OK: ',snr_min)
    reward = snr_min * 1e06
    return reward, state_next, done , theta_all


def environment2(state, action,theta, config):
    state_next = state  + action
    state_next[2] = 5
    state[2] = 5
    done = 0
    snr_all = []
    sensor = 0

    # ---------------channel_state---------------#
    M = config.M
    N = config.N
    # 1.channels
    AP = config.AP
    USER0 = config.USER0
    USER1 = config.USER1
    USER2 = config.USER2
    IRS = state_next
    d_AI0 = math.sqrt((AP[0] - IRS[0]) ** 2 + (AP[1] - IRS[1]) ** 2 + (AP[2] - IRS[2]) ** 2)
    d_IU0 = math.sqrt((IRS[0] - USER0[0]) ** 2 + (IRS[1] - USER0[1]) ** 2 + (IRS[2] - USER0[2]) ** 2)
    d_AU0 = math.sqrt((AP[0] - USER0[0]) ** 2 + (AP[1] - USER0[1]) ** 2 + (AP[2] - USER0[2]) ** 2)

    d_IU1 = math.sqrt((IRS[0] - USER1[0]) ** 2 + (IRS[1] - USER1[1]) ** 2 + (IRS[2] - USER1[2]) ** 2)
    d_AU1 = math.sqrt((AP[0] - USER1[0]) ** 2 + (AP[1] - USER1[1]) ** 2 + (AP[2] - USER1[2]) ** 2)

    d_IU2 = math.sqrt((IRS[0] - USER2[0]) ** 2 + (IRS[1] - USER2[1]) ** 2 + (IRS[2] - USER2[2]) ** 2)
    d_AU2 = math.sqrt((AP[0] - USER2[0]) ** 2 + (AP[1] - USER2[1]) ** 2 + (AP[2] - USER2[2]) ** 2)

    # 2. channel propogation, log distance model
    d_ref = config.d_ref  # reference distance
    exponent_AU = config.exponent_AU  # degradation exponent
    exponent_AI = config.exponent_AI  # degradation exponent
    exponent_IU = config.exponent_IU  # degradation exponent
    L0 = config.L0  # pass loss at the reference distance
    gain = config.gain
    Ld = config.Ld  # extra loss at direct link

    phase_g, phase_H, phase_f , phase_g1, phase_f1, phase_g2, phase_f2 = phase_from_subset(M, N)

    amplify_H = LogdistancePropagation(d_AI0, d_ref, L0 - gain, exponent_AI)
    amplify_g0 = LogdistancePropagation(d_AU0, d_ref, L0 + Ld, exponent_AU)
    amplify_f0 = LogdistancePropagation(d_IU0, d_ref, L0, exponent_IU)

    amplify_g1 = LogdistancePropagation(d_AU1, d_ref, L0 + Ld, exponent_AU)
    amplify_f1 = LogdistancePropagation(d_IU1, d_ref, L0, exponent_IU)

    amplify_g2 = LogdistancePropagation(d_AU2, d_ref, L0 + Ld, exponent_AU)
    amplify_f2 = LogdistancePropagation(d_IU2, d_ref, L0, exponent_IU)

    H = amplify_H * phase_H
    g0 = amplify_g0 * phase_g
    f0 = amplify_f0 * phase_f

    g1 = amplify_g1 * phase_g1
    f1 = amplify_f1 * phase_f1

    g2 = amplify_g2 * phase_g2
    f2 = amplify_f2 * phase_f2

    rho = config.rho

    # ---------------CVX求theta---------------#
    for i in range(3):
        if i ==0 :
            H_f = H @ np.diag(f0.reshape(-1))
            H_f_mat = mat(H_f)
            g_mat = mat(g0)
        elif i == 1:
            H_f = H @ np.diag(f1.reshape(-1))
            H_f_mat = mat(H_f)
            g_mat = mat(g1)
        elif i==2:
            H_f = H @ np.diag(f2.reshape(-1))
            H_f_mat = mat(H_f)
            g_mat = mat(g2)
        theta_mat = mat(theta[i]).T

        # ---------------HAP端的波束形成w---------------#

        w = np.array([[complex(10, -5)], [complex(10, -5)]])# 3.23685584 - 2.47857043 4.12022957 0.01185447
        w_mat = mat(w)

        #-------------AUV_station - -----------#
        area_flag = 0
        if (state_next[0] < 0 ) | (state_next[0] > 20):
            area_flag = 1
        if (state_next[1] < -10) | (state_next[1] > 10):
            area_flag = 1
        if (state_next[2] < 0) | (state_next[2] > 20):
            area_flag=1
        if area_flag:
            return -5, state, done,0
        #-------------snr_min - -----------#
        snr_min_ = (np.linalg.norm(np.conj(g_mat + rho * H_f_mat * theta_mat).T * w_mat, ord=2))#可以直接在matlab中计算，看看是否比这种方式快
        snr_all.append(snr_min_)

    i = argmax(snr_all)

    snr = snr_all[i]

    if i == 0:
        sensor = 0
    elif i ==1:
        sensor = 1
    else:
        sensor = 2



    snr_min = snr
    #-------------energy_min - -----------#
    # energy_min_ = (np.linalg.norm(np.conj(H_mat).T * w_mat, ord=2))
    # energy_min = config.eta * (1 - rho * rho) * energy_min_ * energy_min_

    # ---------------Reward---------------
    # if snr_min < 1e-7:  # SNR约束
    #     print('SNR not OK')
    #     # next_state = channel_state(action[0], action[1], action[2])
    #     reward = -3500  # 如果不符合约束，则输出一个绝对值很大的负值。该负值需要小于算法所获得的最小值，基于实验结果进行调试。
    #     state_next = [5, 0, 0]  # 不符合约束，结束探索，返回起点
    #     done = 1
    #     return reward, state_next, done
    # else:
    #     # print('SNR OK: ',snr_min)
    reward = abs(snr) * 1e03
    return reward, state_next, done, sensor



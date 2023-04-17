import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import integrate
from tqdm import tqdm

from parse import *
plt.grid()

la, lb, ls = 4, 6, 18
Ra, Rb = 4, 5
Na, Nb = 2, 2

Q = np.array([0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              ls, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              ls, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, ls, 0, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, ls, 0, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, ls, 0, 0, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 2 * lb, 1 * la, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 2*lb, 2 * la, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 0, 2 * lb, 1 * la, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 2 * lb, 2 * la, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 2 * lb, 1 * la, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 2*lb, 1 * la,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ls, 0, 0, 0, ]).reshape(24, 24)

Q_ = np.diag(Q.sum(axis=1))
Q = Q - Q_

# для латеха
# print(alg_Kolmogorov(Q))
# print(Kolmogorov(Q))
# print(print_matrix(Q))


# Неявный метод Эйлера для решения системы
def backward_euler(u0, tau, vec, Q_T):
    from scipy import optimize
    from scipy.spatial import distance
    t = [0]
    u = [[x for x in u0]]

    def Phi(z, v):
        return z - tau * (Q_T @ z) - v

    u.append(optimize.fsolve(Phi, u[-1], args=(u[-1])))
    t.append(t[-1] + tau)

    # интегрируем пока L2 норма вектора невязки с ранее рассчитанным предельным вектором
    # составляла не более 10% L2 норма последнего
    while distance.euclidean(u[-1], vec) > 0.01 * np.linalg.norm(vec):
        u.append(optimize.fsolve(Phi, u[-1], args=(u[-1])))
        t.append(t[-1] + tau)

    for _ in range(int(t[-1] / tau)):
        u.append(optimize.fsolve(Phi, u[-1], args=(u[-2])))
        t.append(t[-1] + tau)

    return np.array(u), t


def Expected_breakdown(u, t, Q):
    list_term = []
    for i in range(len(Q)):
        if abs(np.min(Q[i])) == ls:
            list_term.append(i)

    breakdown = np.zeros(len(u))

    for i in list_term:
        breakdown += u[:, i]
    plt.plot(t, breakdown, label=r'$1-R_t(t)$', color='olive')
    # mu = sc.integrate.simpson(breakdown, x=t)
    mu = np.max(breakdown)
    plt.plot(t, [mu for _ in t],'--', label=r'$\mathbb{E}[P(1-R_t(t))]$ = %f' % mu, color='red')
    print("математическое ожидание вероятности отказа %.3f " % mu)
    plt.legend()


def Kolmogorov_equations(Q, flag):
    # Транспонирование матрицы интенсивностей переходов
    # для использования при решении системы dp/dt = Q^T * p
    Q_T = np.copy(Q.T)

    # для установившегося режима работы вид системы 0 = Q^T * p
    Q_stat = np.copy(Q_T)
    Q_stat[0] = np.ones(len(Q_T))
    b = np.zeros(len(Q_T))
    b[0] = 1
    p_stat = np.linalg.solve(Q_stat, b)

    print("предельные вероятности ", [round(x,2) for x in p_stat])


    # Решение системы с помощью неявного метода Эйлера
    u0 = [0 for _ in range(len(Q_T))]
    u0[0] = 1
    u, t = backward_euler(u0, 1e-4, p_stat, Q_T)

    plt.ylabel(r'$P$')
    plt.xlabel(r'$t$')
    if flag[1]:
        if flag[0]:
            for i in range(1,len(Q_T)):
                plt.plot(t, u[:, i], label=r'$P_{%d}(t)$' % i)
        else:
            plt.plot(t, u[:, 0], label=r'$P_{%d}(t)$' % 0)

        plt.legend(fontsize=7)
    return u, t, p_stat


# моделирование одного эпизода с непрерывным временем
def MD(m):
    from scipy.spatial import distance

    w_A, w_B = [Ra], [Rb]

    def find_lambda(line):
        if np.sum(line) == ls:
            return [0, 0], [0, 0], [ls, line.index(ls)]

        for i in range(0, len(line)):
            if line[i] > 0:
                if ls in line:
                    return [line[i], i], \
                           [np.max(line[i + 1::]),
                            line.index(np.max(line[i + 1::]))], \
                           [ls, line.index(ls)]

                return [line[i], i], \
                       [np.max(line[i + 1::]),
                        line.index(np.max(line[i + 1::]))], [0, 0]

    def F_t(l, y):
        return -np.log(1 - y) / l

    current_s = 0
    current_t = 0
    states_tr = [current_s]
    t_tr = [0]

    times = np.zeros(len(m))
    last = np.zeros(len(m))

    flag = False  # выход на установившийся режим работы
    t_ust = 0

    while 1:
        l_b, l_a, l_s = find_lambda(m[current_s])
        t_cur_s = F_t(l_a[0] + l_b[0] + l_s[0],
                      np.random.uniform(low=0.0, high=1.0, size=None))  # -log(1-y)/(lambda_a+lambda_b)

        times[current_s] += t_cur_s

        current_t += t_cur_s
        idx_b = l_b[1]
        idx_a = l_a[1]
        idx_s = l_s[1]
        current_s = np.random.choice([idx_a, idx_b, idx_s],
                                     p=[l_a[0] / (l_a[0] + l_b[0] + l_s[0]),
                                        l_b[0] / (l_a[0] + l_b[0] + l_s[0]),
                                        l_s[0] / (l_a[0] + l_b[0] + l_s[0])])

        # если починка
        if current_s == idx_s:
            # если одинаково поломались
            if Ra - w_A[-1] == Rb - w_B[-1]:
                w_A.append(w_A[-1] + 1 * (la > lb))
                w_B.append(w_B[-1] + 1 * (lb > la))
            else:
                w_A.append(w_A[-1] + 1 * (Ra - w_A[-1] > Rb - w_B[-1]))
                w_B.append(w_B[-1] + 1 * (Rb - w_B[-1] > Ra - w_A[-1]))
        else:
            # если ломаются
            w_A.append(w_A[-1] - 1 * (current_s == idx_a))
            w_B.append(w_B[-1] - 1 * (current_s == idx_b))

        # для дальнейшей отрисовки
        states_tr.append(current_s)
        t_tr.append(current_t)

        if (distance.euclidean(times / current_t, last) < 0.001) * (not flag):
            flag = True
            t_ust = current_t
            times = np.zeros(len(m))  # сбрасываем
            w_A = w_A[-1::]
            w_B = w_B[-1::]

        if flag * (current_t > t_ust * 2):
            return states_tr, t_tr, [np.mean(w_A), np.mean(w_B)], times / (current_t - t_ust), t_ust, current_t - t_ust

        last = times / current_t


def imitational_modeling(Q, N, f):
    plt.ylabel('S')
    plt.xlabel('t')
    plt.yticks(np.arange(0, len(Q), step=1))
    stat_w_a_b = []
    stat_t_term = []
    stat_p = []
    stat_repair = []
    for _ in tqdm(range(N)):
        s_tr, t_tr, stat_w, times, t_ust, t_model = f(Q)
        stat_w_a_b.append([stat_w[0], stat_w[1]])
        stat_t_term.append(t_ust)
        stat_p.append(times)
        stat_repair.append((t_model - times[0]) / t_model)
        plt.plot(t_tr, s_tr, 'o--')
        # print_vec(times)
        # print(t_term)

    print("коэффициент загрузки ремонтной службы %.3f" % np.mean(stat_repair))
    print("среднее число готовых к эксплуатации устройств типа  A и B ",
          [np.round(x, 2) for x in np.mean(stat_w_a_b, axis=0)])
    print("среднее t выхода на установившийся режим работы %.3f" % np.mean(stat_t_term))
    print("статистические оценки предельных вероятностей после выхода на установившийся режим \n",
          [np.round(x, 2) for x in np.mean(stat_p, axis=0)])


u, t, p_pred = Kolmogorov_equations(Q, [0, 0])

Expected_breakdown(u, t, Q)

# print("вероятности")
imitational_modeling(Q.tolist(), 100, MD)
# plt.legend()
plt.show()

from Latex.latex import make_latex
import pydot
import dot2tex
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import modeling

name, var, g = "Тэн Марина Миновна",	169,	5
n = var
name_short = name.split(" ")[0] + " " + name.split(" ")[1][0] + ". " + name.split(" ")[2][0] + "."

T = 0.5
dt = 0.001
N = 100


def graph(Ra, Rb, Na, Nb, La, L_b):
    # GREEN STATE
    graph = pydot.Dot("my_graph", graph_type="digraph", bgcolor="yellow")
    graph.set_strict(True)
    i = 0

    # Start node
    graph.add_node(pydot.Node("s" + str(Na) + str(Ra) + str(Nb) + str(Rb),
                                  texlbl="$S^{" + str(Na) + str(Ra) + "}_{" + str(Nb) + str(Rb) + "}$",
                                  style="filled", fillcolor="green", group=i))

    now_line = ["s" + str(Na) + str(Ra) + str(Nb) + str(Rb)]

    paths = dict()
    paths["s" + str(Na) + str(Ra) + str(Nb) + str(Rb)] = [0, 0, 0]
    p = 1

    while len(now_line) > 0:
        prev_line = now_line
        now_line = []
        i = i + 1

        for node in prev_line:
            nNa = int(node[1])
            nRa = int(node[2])
            nNb = int(node[3])
            nRb = int(node[4])
            if nRb > 0:
                if "s" + str(nNa) + str(nRa) + str(nNb) + str(nRb - 1) not in now_line:
                    graph.add_node(pydot.Node("s" + str(nNa) + str(nRa) + str(nNb) + str(nRb - 1),
                                                  texlbl="$S^{" + str(nNa) + str(nRa) + "}_{" + str(nNb) + str(nRb - 1) + "}$",
                                                  group=i))
                    # i = i + 1

                    paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb - 1)] = [p,
                        paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][1],
                        paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][2] + 1]
                    p = p + 1

                    now_line.append("s" + str(nNa) + str(nRa) + str(nNb) + str(nRb - 1))
                graph.add_edge(pydot.Edge("s" + str(nNa) + str(nRa) + str(nNb) + str(nRb),
                                          "s" + str(nNa) + str(nRa) + str(nNb) + str(nRb - 1),
                                          color="blue", label=" ", texlbl=f"${nNb}\\lambda_B$", len=1, weight=10))
            else:
                graph.add_node(pydot.Node("s" + str(nNa) + str(nRa) + str(nNb - 1) + str(nRb),
                                              texlbl="$S^{" + str(nNa) + str(nRa) + "}_{" + str(nNb - 1) + str(nRb) + "}$",
                                              style="filled", fillcolor="red", group=i))
                # i = i + 1

                paths["s" + str(nNa) + str(nRa) + str(nNb - 1) + str(nRb)] = [-p,
                    paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][1],
                    paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][2] + 1]
                p = p + 1

                graph.add_edge(pydot.Edge("s" + str(nNa) + str(nRa) + str(nNb) + str(nRb),
                                          "s" + str(nNa) + str(nRa) + str(nNb - 1) + str(nRb),
                                          color="blue", label=" ", texlbl=f"${nNb}\\lambda_B$", len=1, weight=10))

            if nRa > 0:
                if "s" + str(nNa) + str(nRa - 1) + str(nNb) + str(nRb) not in now_line:
                    graph.add_node(pydot.Node("s" + str(nNa) + str(nRa - 1) + str(nNb) + str(nRb),
                                                  texlbl="$S^{" + str(nNa) + str(nRa - 1) + "}_{" + str(nNb) + str(nRb) + "}$",
                                                  group=i))
                    # i = i + 1

                    paths["s" + str(nNa) + str(nRa - 1) + str(nNb) + str(nRb)] = [
                        p,
                        paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][1] + 1,
                        paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][2]]
                    p = p + 1

                    now_line.append("s" + str(nNa) + str(nRa - 1) + str(nNb) + str(nRb))
                graph.add_edge(pydot.Edge("s" + str(nNa) + str(nRa) + str(nNb) + str(nRb),
                                          "s" + str(nNa) + str(nRa - 1) + str(nNb) + str(nRb),
                                          color="blue", label=" ", texlbl=f"${nNa}\\lambda_A$", len=1, weight=10))
            else:
                if nNa > 1:
                    if "s" + str(nNa - 1) + str(nRa) + str(nNb) + str(nRb) not in now_line:
                        graph.add_node(pydot.Node("s" + str(nNa - 1) + str(nRa) + str(nNb) + str(nRb),
                                                      texlbl="$S^{" + str(nNa - 1) + str(nRa) + "}_{" + str(nNb) + str(nRb) + "}$",
                                                      group=i))
                        # i = i + 1

                        paths["s" + str(nNa - 1) + str(nRa) + str(nNb) + str(nRb)] = [
                            p,
                            paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][1] + 1,
                            paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][2]]
                        p = p + 1

                        now_line.append("s" + str(nNa - 1) + str(nRa) + str(nNb) + str(nRb))
                    graph.add_edge(pydot.Edge("s" + str(nNa) + str(nRa) + str(nNb) + str(nRb),
                                              "s" + str(nNa - 1) + str(nRa) + str(nNb) + str(nRb),
                                              color="blue", label=" ", texlbl=f"${nNa}\\lambda_A$", len=1, weight=10))
                else:
                    graph.add_node(pydot.Node("s" + str(nNa - 1) + str(nRa) + str(nNb) + str(nRb),
                                              texlbl="$S^{" + str(nNa - 1) + str(nRa) + "}_{" + str(nNb) + str(nRb) + "}$",
                                              style="filled", fillcolor="red", group=i))
                    # i = i + 1
                    paths["s" + str(nNa - 1) + str(nRa) + str(nNb) + str(nRb)] = [
                        -p,
                        paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][1] + 1,
                        paths["s" + str(nNa) + str(nRa) + str(nNb) + str(nRb)][2]]
                    p = p + 1
                    graph.add_edge(pydot.Edge("s" + str(nNa) + str(nRa) + str(nNb) + str(nRb),
                                              "s" + str(nNa - 1) + str(nRa) + str(nNb) + str(nRb),
                                              color="blue", label=" ", texlbl=f"${nNa}\\lambda_A$", len=1, weight=10))

    for node in graph.get_nodes():
        nNa = int(node.get_name()[1])
        nRa = int(node.get_name()[2])
        nNb = int(node.get_name()[3])
        nRb = int(node.get_name()[4])
        s = []

        for edge in graph.get_edges():
            if edge.get_destination() == node.get_name():
                s.append([edge.get_source(), edge.get_attributes()["texlbl"]])

        if len(s) == 1:
            graph.add_edge(pydot.Edge(node.get_name(), s[0][0], color="green",
                                         label=" ", texlbl=f"$\\lambda_S$", weight=1))
        elif len(s) == 2:
            if paths[node.get_name()][1] > paths[node.get_name()][2] or \
                    (paths[node.get_name()][1] == paths[node.get_name()][2] and l_A * int(s[1][1][1]) > l_B * int(s[0][1][1])):
                graph.add_edge(pydot.Edge(node.get_name(), s[0][0], color="green",
                                             label=" ", texlbl=f"$\\lambda_S$", len=1, weight=1))
            else:
                graph.add_edge(pydot.Edge(node.get_name(), s[1][0], color="green",
                                             label=" ", texlbl=f"$\\lambda_S$", len=1, weight=1))

    # print(graph.to_string())
    return graph, paths


def graph_to_file(graph):
    print(graph.to_string())
    graph.create_svg("graph.svg")


def graph_ls(graph, paths, la, lb):
    graph_ls = graph
    for node in graph.get_nodes():
        nNa = int(node.get_name()[1])
        nRa = int(node.get_name()[2])
        nNb = int(node.get_name()[3])
        nRb = int(node.get_name()[4])
        s = []

        for edge in graph.get_edges():
            if edge.get_destination() == node.get_name():
                s.append([edge.get_source(), edge.get_attributes()["texlbl"]])

        if len(s) == 1:
            graph_ls.add_edge(pydot.Edge(node.get_name(), s[0][0], color="green",
                                         label=" ", texlbl=f"$\\lambda_S$", weight=1))
        elif len(s) == 2:
            if paths[node.get_name()][1] > paths[node.get_name()][2] or \
                    (paths[node.get_name()][1] == paths[node.get_name()][2] and la * int(s[1][1][1]) > lb * int(s[0][1][1])):
                graph_ls.add_edge(pydot.Edge(node.get_name(), s[0][0], color="green",
                                             label=" ", texlbl=f"$\\lambda_S$", len=1, weight=1))
            else:
                graph_ls.add_edge(pydot.Edge(node.get_name(), s[1][0], color="green",
                                             label=" ", texlbl=f"$\\lambda_S$", len=1, weight=1))
    # print("___________")
    # print(graph_ls.to_string())
    return graph_ls


def matrix(l_A, l_B, l_S, graph_dot):
    edges = graph_dot.get_edges()
    nodes = graph_dot.get_nodes()

    d = np.zeros((len(nodes), len(nodes)))
    l1_pat = re.compile(r"\$.\\lambda_A\$")
    l2_pat = re.compile(r"\$.\\lambda_B\$")
    ls_pat = re.compile(r"\$\\lambda_S\$")

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            for edge in edges:
                if edge.get_source() == nodes[i].get_name() and edge.get_destination() == nodes[j].get_name():
                    if re.match(l1_pat, edge.get_attributes()['texlbl']):
                        d[i][j] = int(edge.get_attributes()['texlbl'][1]) * l_A
                    elif re.match(l2_pat, edge.get_attributes()['texlbl']):
                        d[i][j] = int(edge.get_attributes()['texlbl'][1]) * l_B
                    elif re.match(ls_pat, edge.get_attributes()['texlbl']):
                        d[i][j] = l_S
        d[i][i] = 0 - sum(d[i])
    return d


def kolmogorov_tex(m):
    s = ""
    for i in range(m.shape[0]):
        s = s + "P^\prime_{" + str(i) + "} = "
        for j in range(m.shape[0]):
            if i != j:
                if m[j][i] >= 1:
                    s = s + str(int(m[j][i])) + "P_{" + str(j) + "} (t) +"
                if m[j][i] <= -1:
                    s = s[:-1] + "-" + str(abs(int(m[j][i]))) + "P_{" + str(j) + "} (t) +"
        for j in range(m.shape[0]):
            if i != j:
                if m[i][j] >= 1:
                    s = s[:-1] + "-" + str(abs(int(m[i][j]))) + "P_{" + str(i) + "} (t) +"
                if m[i][j] <= -1:
                    s = s + str(int(m[i][j])) + "P_{" + str(i) + "} (t) +"
        s = s[:-1] + "\\\\ \n"
    s = s[:-4]
    return s


def kolmogorov_0_tex(m):
    s = ""
    for i in range(m.shape[0]):
        s = s + "0 = "
        for j in range(m.shape[0]):
            if i != j:
                if m[j][i] >= 1:
                    s = s + str(int(m[j][i])) + "P_{" + str(j) + "} (t) +"
                if m[j][i] <= -1:
                    s = s[:-1] + "-" + str(abs(int(m[j][i]))) + "P_{" + str(j) + "} (t) +"
        for j in range(m.shape[0]):
            if i != j:
                if m[i][j] >= 1:
                    s = s[:-1] + "-" + str(abs(int(m[i][j]))) + "P_{" + str(i) + "} (t) +"
                if m[i][j] <= -1:
                    s = s + str(int(m[i][j])) + "P_{" + str(i) + "} (t) +"
        s = s[:-1] + "\\\\ \n"
    s = s[:-4]
    return s


def Kolmogorov_equations(Q):
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
        while distance.euclidean(u[-1], vec) > 0.1 * np.linalg.norm(vec):
            u.append(optimize.fsolve(Phi, u[-1], args=(u[-1])))
            t.append(t[-1] + tau)

        for _ in range(int(t[-1] / tau)):
            u.append(optimize.fsolve(Phi, u[-1], args=(u[-2])))
            t.append(t[-1] + tau)

        return np.array(u), t

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

    fig1, ax1 = plt.subplots(figsize=(10, 7), dpi=80)
    fig2, ax2 = plt.subplots(figsize=(10, 7), dpi=80)
    ax1.grid()
    ax1.set_ylabel(r'$P$')
    ax1.set_xlabel(r'$t$')
    ax2.grid()
    ax2.set_ylabel(r'$P$')
    ax2.set_xlabel(r'$t$')

    ax1.plot(t, u[:, 0], label=r'$P_{%d}(t)$' % 0)
    for i in range(1, len(Q_T)):
        ax2.plot(t, u[:, i], label=r'$P_{%d}(t)$' % i)

    ax1.legend(fontsize=7)
    ax2.legend(fontsize=7)

    fig1.savefig('Latex/res/Images/P_o.png')
    fig2.savefig('Latex/res/Images/P_i.png')

    return u, t, p_stat


def expected_breakdown(u, t, Q):
    list_term = []
    for i in range(len(Q)):
        if abs(np.min(Q[i])) == l_S:
            list_term.append(i)

    breakdown = np.zeros(len(u))

    for i in list_term:
        breakdown += u[:, i]

    fig3, ax3 = plt.subplots(figsize=(10, 7), dpi=80)
    ax3.grid()
    ax3.set_ylabel(r'$P$')
    ax3.set_xlabel(r'$t$')
    ax3.plot(t, breakdown, label=r'$1-R_t(t)$', color='olive')
    # mu = sc.integrate.simpson(breakdown, x=t)
    mu = np.max(breakdown)
    ax3.plot(t, [mu for _ in t],'--', label=r'$\mathbb{E}[P(1-R_t(t))]$ = %f' % mu, color='red')
    print("математическое ожидание вероятности отказа %.3f " % mu)
    fig3.legend()
    fig3.savefig('Latex/res/Images/R_t.png')
    return mu


# def imitational_modeling(Q, N, f):
#     fig4, ax4 = plt.subplots(figsize=(10, 7), dpi=80)
#     ax4.set_ylabel('S')
#     ax4.set_xlabel('t')
#     plt.yticks(np.arange(0, len(Q), step=1))
#     stat_w_a_b = []
#     stat_t_term = []
#     stat_p = []
#     stat_repair = []
#     for _ in tqdm(range(N)):
#         s_tr, t_tr, stat_w, times, t_ust, t_model = f(Q)
#         stat_w_a_b.append([stat_w[0], stat_w[1]])
#         stat_t_term.append(t_ust)
#         stat_p.append(times)
#         stat_repair.append((t_model - times[0]) / t_model)
#         ax4.plot(t_tr, s_tr, 'o--')
#         # print_vec(times)
#         # print(t_term)
#
#     fig4.savefig('Latex/res/Images/term.png')
#     print("коэффициент загрузки ремонтной службы %.3f" % np.mean(stat_repair))
#     # print("среднее число готовых к эксплуатации устройств типа  A и B ",
#     #       [np.round(x, 2) for x in np.mean(stat_w_a_b, axis=0)])
#     # print("среднее t выхода на установившийся режим работы %.3f" % np.mean(stat_t_term))
#     # print("статистические оценки предельных вероятностей после выхода на установившийся режим \n",
#     #       [np.round(x, 2) for x in np.mean(stat_p, axis=0)])
#
#     return np.mean(stat_repair), [np.round(x, 2) for x in np.mean(stat_w_a_b, axis=0)]


# def MD(m):
#     from scipy.spatial import distance
#
#     w_A, w_B = [Ra], [Rb]
#
#     def find_lambda(line):
#         if np.sum(line) == l_S:
#             return [0, 0], [0, 0], [l_S, line.index(l_S)]
#
#         for i in range(0, len(line)):
#             if line[i] > 0:
#                 if l_S in line:
#                     return [line[i], i], \
#                            [np.max(line[i + 1::]),
#                             line.index(np.max(line[i + 1::]))], \
#                            [l_S, line.index(l_S)]
#
#                 return [line[i], i], \
#                        [np.max(line[i + 1::]),
#                         line.index(np.max(line[i + 1::]))], [0, 0]
#
#     def F_t(l, y):
#         return -np.log(1 - y) / l
#
#     current_s = 0
#     current_t = 0
#     states_tr = [current_s]
#     t_tr = [0]
#
#     times = np.zeros(len(m))
#     last = np.zeros(len(m))
#
#     flag = False  # выход на установившийся режим работы
#     t_ust = 0
#
#     while 1:
#         l_b, l_a, l_s = find_lambda(m[current_s])
#         t_cur_s = F_t(l_a[0] + l_b[0] + l_s[0],
#                       np.random.uniform(low=0.0, high=1.0, size=None))  # -log(1-y)/(lambda_a+lambda_b)
#
#         times[current_s] += t_cur_s
#
#         current_t += t_cur_s
#         idx_b = l_b[1]
#         idx_a = l_a[1]
#         idx_s = l_s[1]
#         current_s = np.random.choice([idx_a, idx_b, idx_s],
#                                      p=[l_a[0] / (l_a[0] + l_b[0] + l_s[0]),
#                                         l_b[0] / (l_a[0] + l_b[0] + l_s[0]),
#                                         l_s[0] / (l_a[0] + l_b[0] + l_s[0])])
#
#         # если починка
#         if current_s == idx_s:
#             # если одинаково поломались
#             if Ra - w_A[-1] == Rb - w_B[-1]:
#                 w_A.append(w_A[-1] + 1 * (l_A > l_B))
#                 w_B.append(w_B[-1] + 1 * (l_B > l_A))
#             else:
#                 w_A.append(w_A[-1] + 1 * (Ra - w_A[-1] > Rb - w_B[-1]))
#                 w_B.append(w_B[-1] + 1 * (Rb - w_B[-1] > Ra - w_A[-1]))
#         else:
#             # если ломаются
#             w_A.append(w_A[-1] - 1 * (current_s == idx_a))
#             w_B.append(w_B[-1] - 1 * (current_s == idx_b))
#
#         # для дальнейшей отрисовки
#         states_tr.append(current_s)
#         t_tr.append(current_t)
#
#         if (distance.euclidean(times / current_t, last) < 0.001) * (not flag):
#             flag = True
#             t_ust = current_t
#             times = np.zeros(len(m))  # сбрасываем
#             w_A = w_A[-1::]
#             w_B = w_B[-1::]
#
#         if flag * (current_t > t_ust * 2):
#             return states_tr, t_tr, [np.mean(w_A), np.mean(w_B)], times / (current_t - t_ust), t_ust, current_t - t_ust
#
#         last = times / current_t


# моделирование одного эпизода с непрерывным временем
def MD(m):
    from scipy.spatial import distance

    w_A, w_B = [Ra], [Rb]

    def find_lambda(line):
        if np.sum(line) == l_S:
            return [0, 0], [0, 0], [l_S, line.index(l_S)]

        for i in range(0, len(line)):
            if line[i] > 0:
                if l_S in line:
                    return [line[i], i], \
                           [np.max(line[i + 1::]),
                            line.index(np.max(line[i + 1::]))], \
                           [l_S, line.index(l_S)]

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
                w_A.append(w_A[-1] + 1 * (l_a > l_b))
                w_B.append(w_B[-1] + 1 * (l_b > l_a))
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

        if (distance.euclidean(times / current_t, last) < 0.00001) * (not flag):
            flag = True
            t_ust = current_t
            times = np.zeros(len(m))  # сбрасываем
            w_A = w_A[-1::]
            w_B = w_B[-1::]

        if flag * (current_t > t_ust * 2):
            return states_tr, t_tr, [np.mean(w_A), np.mean(w_B)], times / (current_t - t_ust), t_ust, current_t - t_ust

        last = times / current_t


def imitational_modeling(Q, N, f):
    fig4, ax4 = plt.subplots(figsize=(10, 7), dpi=80)
    ax4.set_ylabel('S')
    ax4.set_xlabel('t')
    ax4.grid()
    stat_w_a_b = []
    stat_t_term = []
    stat_p = []
    stat_repair = []
    for i in tqdm(range(N)):
        s_tr, t_tr, stat_w, times, t_ust, t_model = f(Q)
        stat_w_a_b.append([stat_w[0], stat_w[1]])
        stat_t_term.append(t_ust)
        stat_p.append(times)
        stat_repair.append((t_model - times[0]) / t_model)
        if i < 1:
            ax4.plot(t_tr, s_tr, 'o--')
        # print_vec(times)
        # print(t_term)

    fig4.savefig('Latex/res/Images/term.png')
    print("коэффициент загрузки ремонтной службы %.3f" % np.mean(stat_repair))
    print("среднее число готовых к эксплуатации устройств типа  A и B ",
          [np.round(x, 2) for x in np.mean(stat_w_a_b, axis=0)])
    print("среднее t выхода на установившийся режим работы %.3f" % np.mean(stat_t_term))
    print("статистические оценки предельных вероятностей после выхода на установившийся режим \n",
          [np.round(x, 2) for x in np.mean(stat_p, axis=0)])

    return np.mean(stat_repair), [np.round(x, 2) for x in np.mean(stat_w_a_b, axis=0)], np.mean(stat_t_term), [np.round(x, 2) for x in np.mean(stat_p, axis=0)]


if __name__ == "__main__":
    # Initialisation
    l_A = g + (n % 3)
    l_B = g + (n % 5)
    Na = 2 + (g % 2)
    # Na = 1
    Nb = 2 + (n % 2)
    # Nb = 2
    Ra = 4 + (g % 2)
    # Ra = 2
    Rb = 5 - (g % 2)
    # Rb = 3
    l_S = (Na + Nb - (g % 3)) * (g + (n % 4))

    graph_dot_ls, paths = graph(Ra - Na, Rb - Nb, Na, Nb, l_A, l_B)
    # graph_dot_ls = graph_ls(graph_dot, paths, l_A, l_B)
    print(graph_dot_ls.to_string())
    graph_tex = dot2tex.dot2tex(graph_dot_ls.to_string(), texmode="PSTricks", figonly=True, prog="circo")
    graph_to_file(graph_dot_ls)

    matrix_np = matrix(l_A, l_B, l_S, graph_dot_ls)
    matrix = matrix_np.tolist()

    u, t, p_pred = Kolmogorov_equations(matrix_np)

    mu = expected_breakdown(u, t, matrix_np)
    kzrs, snge, t_sr_v, p_pred_exp = imitational_modeling(matrix, 100, MD)

    t_sr_v_dsm, p_pred_exp_dsm, modeling_log = modeling.imitational_modeling(100, l_A, l_B, l_S, Ra, Rb, Na, Nb, pr=False)

    make_latex(var, g, name, name_short, l_A, l_B, l_S, Na, Nb, Ra, Rb, graph_tex, graph_dot_ls, matrix_np,
               kolmogorov_tex(matrix_np), kolmogorov_0_tex(matrix_np), p_pred, mu, kzrs, snge, t_sr_v, p_pred_exp,
               t_sr_v_dsm, p_pred_exp_dsm, modeling_log)

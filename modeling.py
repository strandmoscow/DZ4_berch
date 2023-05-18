from tqdm import tqdm
import numpy as np
from scipy.spatial import distance

np.random.seed(0)
eps = 0.005

class Unit:
    def __init__(self, typeUnit, lambda_breakdown, lambda_repair):
        self.typeUnit = typeUnit
        self.lambda_breakdown = lambda_breakdown
        self.lambda_repair = lambda_repair

    def time_breakdown(self):
        return -1. / self.lambda_breakdown * np.log(1-np.random.uniform(low=0.0, high=1.0, size=None))

    def time_repair(self):
        return -1. / self.lambda_repair * np.log(1-np.random.uniform(low=0.0, high=1.0, size=None))

    def get_typeID(self):
        return self.typeUnit


def discrete_event_based(la, lb, ls, Ra, Rb, Na, Nb, pr, i, s):
    events = []
    cur_pos = 0
    cur_time = 0
    broken_A, broken_B = 0, 0

    def make_event(pos, A_working, A_stock, A_broken, B_working, B_stock, B_broken, broken_A, broken_B, pr, i):
        s = ''
        event = events[pos][1]
        if i == 0:
            s = s + str(event) + " в момент времени %.3f " % events[pos][0] + "всего устройств типа А и В " + str(len(A_working)+len(A_stock)) + " " + str(len(B_working)+len(B_stock)) + '\n'
        if pr:
            print(event, "в момент времени %.3f" % events[pos][0], "всего устройств типа А и В ", str(len(A_working)+len(A_stock)), " ", str(len(B_working)+len(B_stock)))

        if event == 'Сломан A':
            if len(A_working) == 1:
                events.pop(pos)
                if i == 0:
                    s = s + " " * 50 + "--------------> пропуск события ввиду неработоспособности системы\n"
                if pr:
                    print(" " * 50 + "--------------> пропуск события ввиду неработоспособности системы", )
                return A_working, A_stock, A_broken, B_working, B_stock, B_broken, broken_A, broken_B, s

            broken_A += 1
            if len(A_stock) > 0:
                A_broken.append(A_stock[0])
                A_stock.pop(0)
            elif len(A_working) > 0:
                A_broken.append(A_working[0])
                A_working.pop(0)
            else:
                events.pop(pos)

        elif event == 'Сломан B':
            if len(B_working) == Nb and len(B_stock) == 0:
                events.pop(pos)
                if i == 0:
                    s = s + " " * 50 + "--------------> пропуск события ввиду неработоспособности системы\n"
                if pr:
                    print(" " * 50 + "--------------> пропуск события ввиду неработоспособности системы",)
                return A_working, A_stock, A_broken, B_working, B_stock, B_broken, broken_A, broken_B, s

            broken_B += 1
            if len(B_stock) > 0:
                B_broken.append(B_stock[0])
                B_stock.pop(0)
            elif len(B_working) > 0:
                B_broken.append(B_working[0])
                B_working.pop(0)
            else:
                events.pop(pos)

        elif event == 'Починен A':
            if len(A_working) == Na and len(A_stock) == Ra-Na:
                events.pop(pos)
                if i == 0:
                    s = s + " " * 50 + "--------------> пропуск события ввиду исправности всех устройств типа А\n"
                if pr:
                    print(" " * 50 + "--------------> пропуск события ввиду исправности всех устройств типа А", )
                return A_working, A_stock, A_broken, B_working, B_stock, B_broken, broken_A, broken_B, s

            broken_A -= 1
            if len(A_working) == Na:
                A_stock.append(Unit('A', la, ls))
            else:
                A_working.append(Unit('A', la, ls))

        elif event == 'Починен B':
            if len(B_working) == Nb and len(B_stock) == Rb-Nb:
                events.pop(pos)
                if i == 0:
                    s = s + " " * 50 + "--------------> пропуск события ввиду исправности всех устройств типа В\n"
                if pr:
                    print(" " * 50 + "--------------> пропуск события ввиду исправности всех устройств типа В", )
                return A_working, A_stock, A_broken, B_working, B_stock, B_broken, broken_A, broken_B, s
            broken_B -= 1
            if len(B_working) == Nb:
                B_stock.append(Unit('B', lb, ls))
            else:
                B_working.append(Unit('B', lb, ls))
        if i == 0:
            s = s + " " * 50 + "--------------> " + str(len(A_working) + len(A_stock)) + " " + str(len(B_working) + len(B_stock)) + "\n"
        if pr:
            print(" " * 50 + "--------------> ", len(A_working) + len(A_stock), " ", len(B_working) + len(B_stock))
        return A_working, A_stock, A_broken, B_working, B_stock, B_broken, broken_A, broken_B, s

    def plan_event(working, broken, cur_time, cur_pos):
        for ev in working + broken:
            if ev in working:
                events.append([ev.time_breakdown()+cur_time, "Сломан" + ' ' + ev.get_typeID()])
            elif ev in broken:
                events.append([ev.time_repair()+cur_time, "Починен" + ' ' + ev.get_typeID()])

        events.sort()
        cur_time += min(events[cur_pos::])[0]
        cur_pos += 1
        return cur_time, cur_pos

    # первичная инициализация
    units_A_working = [Unit('A', la, ls) for _ in range(Na)]
    units_A_stock = [Unit('A', la, ls) for _ in range(Ra-Na)]  # резерв
    units_A_broken = []
    units_B_working = [Unit('B', lb, ls) for _ in range(Nb)]
    units_B_stock = [Unit('B', lb, ls) for _ in range(Rb-Nb)]  # резерв
    units_B_broken = []

    # distance.euclidean(times / current_t, last) < 0.00001
    stat_broken_a_b=[[broken_A, broken_B]]
    last_stat=[[Na, Nb]]

    Flag = False
    t_term = 0
    while 1:
        working = units_A_working + units_B_working
        broken = units_A_broken + units_B_broken
        cur_time, cur_pos = plan_event(working, broken, cur_time, cur_pos)
        units_A_working, units_A_stock, units_A_broken, units_B_working, units_B_stock, units_B_broken, broken_A, broken_B, sl = \
            make_event(cur_pos-1, units_A_working, units_A_stock, units_A_broken, units_B_working, units_B_stock, units_B_broken, broken_A, broken_B, pr, i)
        stat_broken_a_b.append([broken_A, broken_B])
        s = s + sl
        if (distance.euclidean(np.mean(stat_broken_a_b, axis=0), np.mean(last_stat, axis=0)) < eps) * (not Flag):
            Flag = True
            t_term = events[cur_pos][0]
            stat_broken_a_b = [[broken_A, broken_B]]

        if Flag * (events[cur_pos][0] > t_term * 2):
            return events[cur_pos][0], np.mean(stat_broken_a_b, axis=0), s

        last_stat = np.copy(stat_broken_a_b)


def imitational_modeling(N, la, lb, ls, Ra, Rb, Na, Nb, pr=False):
    stat_w_a_b = []
    stat_t_term = []
    s = ""
    for i in tqdm(range(N)):
        t_term, stat_br, s = discrete_event_based(la, lb, ls, Ra, Rb, Na, Nb, pr, i, s)
        # print(stat_br)
        stat_w_a_b.append([Na-stat_br[0]+1, Nb-stat_br[1]+1])
        stat_t_term.append(t_term)

    print("среднее число готовых к эксплуатации устройств типа  A и B ",
          [np.round(x, 2) for x in np.mean(stat_w_a_b, axis=0)])
    print("среднее t выхода на установившийся режим работы %.3f" % np.mean(stat_t_term))
    return [np.round(x, 2) for x in np.mean(stat_w_a_b, axis=0)], np.mean(stat_t_term), s

# imitational_modeling(100)




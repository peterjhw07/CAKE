"""CAKE Get Events"""

import copy
import numpy as np


# Unpacks time events
def get_events(lst):
    event_times = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                if isinstance(sub_item, int) or isinstance(sub_item, float):
                    event_times.append(sub_item)
        elif isinstance(item, int) or isinstance(item, float):
            event_times.append(item)
    return sorted(set(event_times))


# Prepares addition events
def prepare_add(num_spec, add, t):
    if add and t:
        event_t = np.array(get_events(t))
        dvols = np.zeros((len(event_t), num_spec))
        for i, j in enumerate(add):
            if j:
                for k in range(len(j)):
                    found_rows = [index for index, value in enumerate(event_t) if value == t[i][k]]
                    dvols[found_rows[0]:, i] = j[k]
        dvol = np.sum(dvols, axis=1)
    else:
        event_t = np.zeros(0)
        dvols = np.zeros((0, num_spec))
        dvol = np.zeros(0)
    return event_t, dvols, dvol


# Prepares subtraction events
def prepare_sub(sub, t):
    if sub[0]:
        event_t = np.array(t)
        dvol = np.zeros(len(t))
        dvol -= [abs(i) for i in sub]
    else:
        event_t = np.zeros(0)
        dvol = np.zeros(0)
    return event_t, dvol


# Prepares temperature events
def prepare_temp(temp, t):
    if temp and t and temp[0]:
        event_t = np.array(t)
        dtemp = np.array(temp)
    else:
        event_t = np.zeros(0)
        dtemp = np.zeros(0)
    return event_t, dtemp


# Prepares discrete events
def get_disc_temp(num_spec, disc_add_vol, t_disc_add, disc_sub_vol, t_disc_sub):
    t, dvols, dvol = prepare_add(num_spec, disc_add_vol, t_disc_add)
    t_sub, dvol_sub = prepare_sub(disc_sub_vol, t_disc_sub)
    dsub = np.zeros(len(t))

    for i in range(len(t_sub)):
        if t_sub[i] in t:
            dvol[t == t_sub[i]] += dvol_sub[i]
            dsub[t == t_sub[i]] = dvol_sub[i]
        else:
            t = np.append(t, t_sub[i])
            dvols = np.vstack((dvols, np.zeros((1, dvols.shape[1]))))
            dvol = np.append(dvol, dvol_sub[i])
            dsub = np.append(dsub, dvol_sub[i])
    row_sort = t.argsort()
    return t[row_sort], dvols[row_sort], dvol[row_sort], dsub[row_sort]


# Gets final discrete events
def get_disc(t_disc, dvols_disc, dvol_disc, dsub_disc, t_cont, vol_cont):
    vol_disc = copy.deepcopy(vol_cont)
    for i, j in enumerate(t_cont):
        if j in t_disc:
            vol_disc[i] -= dvol_disc[t_disc == j]
        else:
            t_disc = np.append(t_disc, j)
            dvols_disc = np.vstack((dvols_disc, np.zeros((1, dvols_disc.shape[1]))))
            dvol_disc = np.append(dvol_disc, 0)
            dsub_disc = np.append(dsub_disc, 0)
    row_sort = t_disc.argsort()
    return dvols_disc[row_sort], dvol_disc[row_sort], dsub_disc[row_sort], vol_disc


# Gets continuous events and adds discrete events
def get_cont_add_disc(num_spec, cont_add_rate, t_cont_add, event_t_disc, dvol_disc, cont_sub_rate, cont_temp_rate,
                      t_cont_temp):
    t_cont, dvols_cont, dvol_cont = prepare_add(num_spec, cont_add_rate, t_cont_add)
    event_t_temp, dtemp = prepare_temp(cont_temp_rate, t_cont_temp)
    t_cont, dvols_cont, dvol_cont, vol_cont, dtemp = get_cont_add_temp(t_cont, dvols_cont, dvol_cont,
                                                                       event_t_temp, dtemp)
    t_cont_add_add, dvols_cont_add, dvol_cont_add, vol_cont_add, dtemp_add = t_cont, dvols_cont, dvol_cont, vol_cont, dtemp

    for i, j in enumerate(event_t_disc):
        if j in t_cont:
            vol_cont_add[t_cont_add_add == j] += dvol_disc[i]
        else:
            if len(t_cont[t_cont < j]) > 0:
                row_find = t_cont < j
                t_cont_add_add = np.append(t_cont_add_add, j)
                dvols_cont_add = np.vstack((dvols_cont_add, dvols_cont[row_find][-1]))
                dvol_cont_add = np.append(dvol_cont_add, dvol_cont[row_find][-1])
                vol_cont_add = np.append(vol_cont_add, dvol_disc[i])
                dtemp_add = np.append(dtemp_add, dtemp[row_find][-1])
            else:
                t_cont_add_add = np.append(t_cont_add_add, j)
                dvols_cont_add = np.vstack((dvols_cont_add, np.zeros((1, dvols_cont.shape[1]))))
                dvol_cont_add = np.append(dvol_cont_add, 0)
                vol_cont_add = np.append(vol_cont_add, dvol_disc[i])
                dtemp_add = np.append(dtemp_add, 0)
    dvol_cont_add -= cont_sub_rate
    row_sort = t_cont_add_add.argsort()
    return t_cont_add_add[row_sort], dvols_cont_add[row_sort], dvol_cont_add[row_sort], \
           vol_cont_add[row_sort], dtemp_add[row_sort]


# Gets continuous events and adds temperature
def get_cont_add_temp(t_cont, dvols_cont, dvol_cont, t_temp, dtemp):
    t_cont_add, dvols_cont_add, dvol_cont_add, dtemp_add, t_temp_add = t_cont, dvols_cont, dvol_cont, dtemp, t_temp
    vol_cont_add = np.zeros(len(t_cont))

    for i, j in enumerate(t_temp):
        if j not in t_cont:
            if len(t_cont[t_cont < j]) > 0:
                row_find = t_cont < j
                t_cont_add = np.append(t_cont_add, j)
                dvols_cont_add = np.vstack((dvols_cont_add, dvols_cont[row_find][-1]))
                dvol_cont_add = np.append(dvol_cont_add, dvol_cont[row_find][-1])
                vol_cont_add = np.append(vol_cont_add, 0)
            else:
                t_cont_add = np.append(t_cont_add, j)
                dvols_cont_add = np.vstack((dvols_cont_add, np.zeros((1, dvols_cont.shape[1]))))
                dvol_cont_add = np.append(dvol_cont_add, 0)
                vol_cont_add = np.append(vol_cont_add, 0)
    for i, j in enumerate(t_cont_add):
        if len(t_temp[t_temp < j]) > 0:
            t_temp_add = np.append(t_temp_add, j)
            dtemp_add = np.vstack((dvols_cont_add, dvols_cont[t_cont < j][-1]))
        else:
            t_temp_add = np.append(t_temp_add, j)
            dtemp_add = np.append(dtemp_add, 0)
    row_sort = t_cont_add.argsort()
    row_sort_temp = t_temp_add.argsort()
    return t_cont_add[row_sort], dvols_cont_add[row_sort], dvol_cont_add[row_sort], vol_cont_add[row_sort], \
           dtemp_add[row_sort_temp]


# Adds row to the start or end
def add_start_end_row(t, dvols, dvol, vol, t_end, cont_sub_rate, dtemp):
    if 0 not in t:
        t = np.append(0, t)
        dvols = np.vstack((np.zeros((1, dvols.shape[1])), dvols))
        dvol = np.append(-cont_sub_rate, dvol)
        vol = np.append(0, vol)
        dtemp = np.append(0, dtemp)
    if t_end not in t:
        t = np.append(t, t_end)
        dvols = np.vstack((dvols, dvols[-1]))
        dvol = np.append(dvol, dvol[-1])
        vol = np.append(vol, 0)
        dtemp = np.append(dtemp, dtemp[-1])
    return t, dvols, dvol, vol, dtemp


# Finally calculates volume
def calc_vol(t, dvol, vol, vol0):
    vol[0] += vol0
    for i in range(1, len(t)): vol[i] += vol[i - 1] + ((t[i] - t[i - 1]) * dvol[i - 1])
    return vol


# Calculates temperature
def calc_temp(t, dtemp, temp0):
    temp = np.zeros(len(dtemp))
    temp[0] = temp0
    for i in range(1, len(t)): temp[i] += temp[i - 1] + ((t[i] - t[i - 1]) * dtemp[i - 1])
    return temp


# Splits t for event calculations
def split_t(t, t_event):
    t_split = [t[(t >= t_event[i]) & (t <= t_event[i + 1])] for i in range(len(t_event) - 2)]
    t_split.append(t[(t >= t_event[-2]) & (t <= t_event[-1])])
    return t_split


# Event data object
class EventData:
    def __init__(self, t, dmol, dvol, dsub, vol, t_dtemp, temp):
        self.t = t
        self.dmol = dmol
        self.dvol = dvol
        self.dsub = dsub
        self.vol = vol
        self.t_dtemp = t_dtemp
        self.temp = temp


# Calculates additions and subtractions of species
def get_conc_events(t, num_spec, vol0, add_sol_conc, cont_add_rate, t_cont_add, disc_add_vol, t_disc_add,
                    cont_sub_rate, disc_sub_vol, t_disc_sub, temp0, cont_temp_rate, t_cont_temp):

    cont_sub_rate = abs(cont_sub_rate) if cont_sub_rate else 0
    t_disc, ddisc_vols, ddisc_vol, ddisc_sub = get_disc_temp(num_spec, disc_add_vol, t_disc_add, disc_sub_vol,
                                                             t_disc_sub)
    t_cont_add, dcont_vols, dcont_vol, cont_vol, t_dcont_temp = get_cont_add_disc(num_spec, cont_add_rate, t_cont_add,
                                                                                  t_disc, ddisc_vol, cont_sub_rate,
                                                                                  cont_temp_rate, t_cont_temp)
    dcont_sub = -cont_sub_rate

    t_event, dcont_vols, dcont_vol, cont_vol, t_dcont_temp = add_start_end_row(t_cont_add, dcont_vols, dcont_vol,
                                                                               cont_vol, t[-1], cont_sub_rate,
                                                                               t_dcont_temp)
    cont_vol = calc_vol(t_event, dcont_vol, cont_vol, vol0)
    cont_temp_rate = calc_temp(t_event, t_dcont_temp, temp0)
    ddisc_vols, ddisc_vol, ddisc_sub, vol_disc = get_disc(t_disc, ddisc_vols, ddisc_vol, ddisc_sub, t_event, cont_vol)

    dmol_disc, dmol_cont = ddisc_vols, dcont_vols
    for i, conc in enumerate(add_sol_conc):
        if conc:
            dmol_disc[:, i] *= conc
            dmol_cont[:, i] *= conc

    disc_event = EventData(t_event, dmol_disc, ddisc_vol, ddisc_sub, vol_disc, [], [])
    cont_event = EventData(t_event, dmol_cont, dcont_vol, dcont_sub, cont_vol, t_dcont_temp, cont_temp_rate)
    t_split = split_t(t, t_event)

    return t_split, disc_event, cont_event

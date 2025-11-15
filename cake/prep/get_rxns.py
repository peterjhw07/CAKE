"""CAKE Get Reaction Parameters from Rxns"""

import numpy as np


# Get reaction mechanism parameters
def get_rxns(rxns):
    if isinstance(rxns, str): rxns = [rxns]
    if isinstance(rxns, list) and isinstance(rxns[0], str) and '->' in rxns[0]:
        rxns = [(tuple(r.replace(' ', '') for r in r.split(' + ')),
                 tuple(p.replace(' ', '') for p in p.split(' + ')))
                for rxn in rxns for r, p in [rxn.split(' -> ', 1)]]

    mod_rxns = []
    for r, p in rxns:
        if isinstance(r, str): r = (r,)
        if isinstance(p, str): p = (p,)
        mod_r, mod_p = [], []
        for i in r:
            if i[0].isdigit():
                num = int(i[0])
                letter = i[1:]
                mod_r.extend([letter] * num)
            else:
                mod_r.append(i)
        for i in p:
            if i[0].isdigit():
                num = int(i[0])
                letter = i[1:]
                mod_p.extend([letter] * num)
            else:
                mod_p.append(i)
        mod_rxns.append((mod_r, mod_p))

    # Determine unique species
    spec = []
    for r, p in mod_rxns:
        for spec_name in r + p:
            spec.append(spec_name)
    spec = [s for i, s in enumerate(spec) if spec.index(s) == i]

    # Determine the stoichiometries of reactions and locations of non-zero values
    stoich, stoich_loc = np.zeros((len(mod_rxns), len(spec))), []
    for k, i in enumerate(spec):
        spec_rxn_indices_i = []
        for j, (r, p) in enumerate(mod_rxns):
            r_count, p_count = r.count(i), p.count(i)
            if p_count - r_count != 0:
                stoich[j, k] = p_count - r_count
                spec_rxn_indices_i.append(j)
        stoich_loc.append(tuple(spec_rxn_indices_i))

    # Determine the orders of reaction and the locations of non-zero values
    ord, ord_loc = np.zeros((len(mod_rxns), len(spec))), []
    for j, (r, _) in enumerate(mod_rxns):
        rxns_r_i = []
        for i in [spec.index(spec_name) for spec_name in r]:
            if i not in rxns_r_i:
                rxns_r_i.append(i)
                ord[j, i] = 1
            else:
                index = rxns_r_i.index(i)
                ord[j, index] += 1
        ord_loc.append(tuple(rxns_r_i))

    return spec, stoich, stoich_loc, ord, ord_loc


if __name__ == "__main__":
    rxns = []
    rxns += (['2A', 'B'], ['C'])
    rxns += (['C'], ['D'])
    rxns += (['C'], ['E'])
    rxns += (['D', 'A', 'E'], ['D', '2F'])
    rxns += (['G', 'C'], ['G', 'A'])

    spec, stoich, stoich_loc, ord, ord_loc = get_rxns(rxns)

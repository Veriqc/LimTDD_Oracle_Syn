from caterpillar_bindings import synthesize_stg_from_tt_tcost, synthesize_stg_from_tt_cxcost
'''
size = 9
mask = (1 << 64) - 1
with open(f'data/exorcism_rand_tcost_{size}.txt', 'w') as f: 
    for tt_num in range(80000):
        print(tt_num)
        tt, tt_tmp = [], tt_num
        while tt_tmp:
            tt.append(tt_tmp & mask)
            tt_tmp >>= 64
        tcost = synthesize_stg_from_tt_tcost(size, 0, tt)
        f.write(f'{tt_num}:{tcost}\n')
'''
size = 8
mask = (1 << 64) - 1
lines = {}
with open(f'data/exorcism_rand_tcost_{size}.txt', 'r') as fin:
    lines = {int(tt): int(cost) for line in fin for tt, cost in (line.strip().split(':'),)}
with open(f'data/exorcism_rand_cxcost_{size}.txt', 'w') as fout:
    counter = 0
    for tt_num in lines.keys():
        print(counter)
        counter += 1

        tt, tt_tmp = [], tt_num
        while tt_tmp:
            tt.append(tt_tmp & mask)
            tt_tmp >>= 64

        cxcost = synthesize_stg_from_tt_cxcost(size, 0, tt)
        fout.write(f'{tt_num}:{cxcost}\n')
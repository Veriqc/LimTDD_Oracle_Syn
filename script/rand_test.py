import random
from caterpillar_bindings import synthesize_stg_from_tt_tcost

size = 11
num = 60000
mask = (1 << 64) - 1
with open(f'data/exorcism_rand_tcost_{size}.txt', 'a') as f: 
    for counter in range(num):
        tt_num = random.randint(0, (1 << (1 << size)) - 1)
        print(counter)
        tt, tt_tmp = [], tt_num
        while tt_tmp:
            tt.append(tt_tmp & mask)
            tt_tmp >>= 64
        tcost = synthesize_stg_from_tt_tcost(size, 0, tt)
        f.write(f'{tt_num}:{tcost}\n')
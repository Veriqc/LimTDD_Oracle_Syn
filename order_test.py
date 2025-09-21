import numpy as np
from TDD.TDD import TDD, Ini_TDD,Clear_TDD,set_index_order,get_unique_table_num,set_root_of_unit,get_count,cont,renormalize,Slicing2,Slicing
from TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs,gen_cir
from TDD.TN import Index,Tensor,TensorNetwork
from TDD.Syn import *

import copy
from qiskit.circuit.library.standard_gates import HGate,U1Gate,U3Gate
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.circuit import CircuitInstruction

to_test = False

import warnings

warnings.simplefilter("ignore")

def oracle_syn2(tdd,n=0,cond={},data_q = -1,qa_used=False,n_r=0):

    if n==0:
        n = tdd.node.key+1
    cir = Circuit(n+2,[])
    qa = n
    data_q = n+1
    
    if tdd.node.key==-1:
        if abs(tdd.weight)<1e-10:
            cir = Circuit(n_r+2,[])
            return cir
        else:
            if qa_used:
                cir.data.append(Gate('x',{qa:1},data_q))
            else:
                cir.data.append(Gate('x',{},data_q))
        return cir
        
    u = tdd.node

    if u.successor[0]!=u.successor[1]:
        #把1分支关闭
        qa_used = True
        cond_n = copy.copy(cond)
        cond_n[u.key] = 1
        g = Gate('x',cond_n,qa)
        g.is_edge_op = True
        cir.data.append(g)

        bran_tdd = get_branch_dd(u,0)
        cir_end_t0 = oracle_syn2(bran_tdd,n,cond|{u.key:0},data_q,qa_used)
        cir.data=cir.data+cir_end_t0.data
    
        #把0分支关闭，1分支打开
        g=Gate('x',cond,qa)
        g.is_edge_op = True
        cir.data.append(g)

        reverse_q = []
        the_map = u.out_maps[1]
        while the_map.level>-1:
            idx = the_map.level
            q = int(tdd.key_2_index[idx][1:])
            reverse_q.append(q)
            the_map=the_map.father
            
        bran_tdd = get_branch_dd(u,1)
        cir_end_t1 = oracle_syn2(bran_tdd,n,cond|{u.key:1},data_q,qa_used)
        for g in cir_end_t1.data:
            for k in g.q_c:
                if k in reverse_q:
                    g.q_c[k] = 1-g.q_c[k]        
        cir.data=cir.data+cir_end_t1.data

        #把0分支再打开
        cond_n = copy.copy(cond)
        cond_n[u.key] = 0
        g=Gate('x',cond_n,qa)
        g.is_edge_op = True
        cir.data.append(g)
    else:
        op_gates = []
        the_map = u.out_maps[1]
        while the_map.level>-1:
            q1 = u.key
            idx = the_map.level
            q = int(tdd.key_2_index[idx][1:])
            g=Gate('x',{q1:1},q)
            g.is_edge_op = True
            op_gates.append(g)
            the_map=the_map.father
        cir.data+=op_gates
        bran_tdd = get_branch_dd(u,0)
        cir_end_t0 = oracle_syn2(bran_tdd,n,cond,data_q,qa_used)
        if abs(u.out_weight[1])<1e-10:
            cir_end_t0 = get_controlled_circuit2(cir_end_t0,{u.key:0},1,True)
        cir.data=cir.data+cir_end_t0.data
        op_gates.reverse()
        cir.data+=op_gates
            
    reverse_q = []
    the_map=tdd.map
    while the_map.level>-1:
        idx=the_map.level
        q = int(tdd.key_2_index[idx][1:])
        reverse_q.append(q)
        the_map=the_map.father 
    for g in cir.data:
        for k in g.q_c:
            if k in reverse_q:
                g.q_c[k] = 1-g.q_c[k]
    return cir

def verify_success(cir,data,n):
    print(data)
    for k in range(2**n):
        input = format(k & ((1 << n) - 1), f'0{n}b')
        current_state = {n-1-q : int(input[q]) for q in range(n)}
        current_state[n] = 1
        current_state[n+1] = 0
        current_state_s = copy.copy(current_state)
        # print(current_state_s)
        for g in cir.data:
            flag = True
            for q in g.q_c:
                if g.q_c[q]!=current_state[q]:
                    flag = False
                    break
            if flag:
                current_state[g.q_t]=1-current_state[g.q_t]
        # print(current_state)
        
        for q in range(len(current_state)-1):
            if current_state[q]!=current_state_s[q]:
                print('Not Successful, see input:',input)
                break                
        if current_state[n+1]!=data[k]:
            print('Not Successful, see input:',input)
        # print('--')
    print('Successful')

t_cost = [0, 0, 7, 16, 24, 62, 80, 200]

def calc_tcost(circ):
    cost = 0
    for gate in circ.data:
        cost += t_cost[len(list(gate.q_c.keys()))]
    return cost

def all_var_orders(num_input):
    import itertools
    orders = []
    for order in list(itertools.permutations(range(num_input))):
        orders.append([f'x{i}' for i in order])
    return orders


def main():
    import sys

    input_num = 4
    if len(sys.argv) >= 2: input_num = int(sys.argv[1])
    lines = []
    with open(f'exorcism_tcost_{input_num}.txt', 'r') as f:
        lines = f.readlines()

    better, eqv, worse = 0, 0, 0
    all = 2 ** (2 ** input_num)
    all_orders = all_var_orders(input_num)
    for tt in range(all):
        print("TT = ", tt)

        tcost_exor = int(lines[tt])
        #print(f"    T cost of exorcism        = {tcost_exor}")

        tcost = 100000
        for order in all_orders:
            tdd,data = get_TDD_from_truth_table(input_num, tt, order)
            circ = oracle_syn2(tdd,n_r=input_num)
            tcost = min(tcost, calc_tcost(circ))
            #print(f"    T cost                    = {tcost}")
            if tcost < tcost_exor: break

        if tcost > tcost_exor: worse += 1
        elif tcost == tcost_exor: eqv += 1
        else: better += 1

    print("****** Comparison of TDD-based Method Alg. 2 to Exorcism  ******")
    print(f"number of input variables: {input_num}")
    print(f"lower  t-cost in {better}({better / all * 100}%) cases")
    print(f"higher t-cost in {worse}({worse / all * 100}%) cases")
    print(f"same   t-cost in {eqv}({eqv / all * 100}%) cases")
    print("****** ******  ******")

if __name__=="__main__":
    main()
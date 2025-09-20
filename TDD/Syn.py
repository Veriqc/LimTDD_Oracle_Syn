from TDD.TDD import TDD, Ini_TDD,Clear_TDD,set_index_order,get_unique_table_num,set_root_of_unit,get_count,cont,renormalize,Slicing2,Slicing
from TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs,gen_cir
from TDD.TN import Index,Tensor,TensorNetwork
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import networkx as nx
import itertools
import copy

to_test = False

class Gate:
    def __init__(self,name='',q_c={},q_t=0,para=np.eye(2)):
        self.name = name
        self.q_c = q_c
        self.q_t = q_t
        # self.nodes = {}
        self.is_edge_op = False
        X = np.array([[0,1],[1,0]])
        if 'x' in name:
            self.para = X
        elif 'p' in name:
            if type(para) == float:
                self.para = np.array([[1,0],[0,np.exp(1j*para)]])
            else:
                self.para = para
        else:
            self.para=para

        if 'u' in name:
            if np.allclose(para,X):
                self.name = 'x'
                self.para = X
            elif abs(para[0][1])<1e-10 and abs(para[1][0])<1e-10:
                self.name = 'p'
        if np.allclose(self.para,np.eye(2)):
            self.name = 'p'
            self.q_c={}
            self.q_t = 0
            
        
    def qubits(self):
        return list(self.q_c.keys())+[self.q_t]
            
    def __eq__(self,other):
        return self.q_c==other.q_c and self.q_t == other.q_t and np.allclose(self.para,other.para)
        
    def __str__(self):
        dict_str = '('+", ".join([f"{key}: {value}" for key, value in self.q_c.items()])+')'
        return str((self.name,dict_str,self.q_t,self.para))
        
    def to_qis_cir(self,n):
        cir = QuantumCircuit(n)
        if 'p' in self.name:
            theta = np.angle(self.para[1][1])
            cir.p(theta,self.q_t)
        elif 'x' in self.name:
            cir.x(self.q_t)
        elif 'u' in self.name:
            u3 = UnitaryGate(self.para)
            cir.append(u3,[self.q_t])
        if len(self.q_c)>0:
            cir_new = QuantumCircuit(n)
            sorted_q_c = dict(sorted(self.q_c.items(), reverse=True))
            ctrl_cond = ''
            new_qubits = []
            for k in sorted_q_c:
                ctrl_cond+=str(sorted_q_c[k]) 
                new_qubits.append(k)
            new_qubits.append(self.q_t)
            # print(ctrl_cond,new_qubits)
            for g, qargs, cargs in cir.data:
                cg = g.control(len(self.q_c),ctrl_state=''.join(reversed(ctrl_cond)))
                cir_new.append(cg,new_qubits)
                return cir_new
        return cir

class Circuit:
    def __init__(self,num_qubits=1,data=[]):
        self.num_qubits = num_qubits
        self.data = data
    def __str__(self):
        s = '['
        for g in self.data:
            s+=str(g)
            s+=','
        s+=']'
        return s      
    def to_qis_cir(self):
        cir = QuantumCircuit(self.num_qubits)
        for g in self.data:
            cir=cir&g.to_qis_cir(self.num_qubits)
        return cir
        
def get_controlled_circuit2(cir,cond = {0:1},add_qubits_num = 0,pass_edge_op=False):
    cir.num_qubits += add_qubits_num
    new_data = []
    for g in cir.data:
        if pass_edge_op and g.is_edge_op:
            continue
        for c in cond:
            g.q_c[c]=cond[c]
    return cir



def simulate(cir,ini=False,n=0):
    tn,all_indexs = cir_2_tn(cir)
    tn.tensors=[ts for ts in tn.tensors if ts.name!='nu_q']
    # n = get_real_qubit_num(cir)
    if n!=0:
        add_inputs(tn,[0]*n,n)

    if ini:
        var=[]
        for idx in all_indexs:
            if idx[0]=='x' and not '_' in idx:
                var.append('a'+idx[1:])
            var.append(idx)
            if idx[0]=='y' and not '_' in idx:
                var.append('b'+idx[1:])
#         print(var)
        for k1 in range(n):
            if not 'x'+str(k1) in var:
                var.append('x'+str(k1))
            for k in range(5*len(all_indexs)):
                s ='x'+str(k1)+'_'+str(k)
                if not s in var:
                    var.append(s)
                s ='y'+str(k1)+'_'+str(k)
                if not s in var:
                    var.append(s)               
            if not 'y'+str(k1) in var:
                var.append('y'+str(k1))
        var.reverse()
        Ini_TDD(index_order=var)
        set_root_of_unit(2**3)
    tdd=tn.cont()
    return tdd

def change_index(tdd,x,y):
    """change the index from x to y"""
    
    idx_set = []
    key_2_index  = dict()
    index_2_key = dict()
    
    for idx in tdd.index_set:
        if idx.key[0]==x:
            idx_set.append(Index(y+idx.key[1:],idx.idx))
        else:
            idx_set.append(idx)
    
    for k in tdd.key_2_index:
        if isinstance(tdd.key_2_index[k],str) and tdd.key_2_index[k][0]==x:
            key_2_index[k]=y+tdd.key_2_index[k][1:]
        else:
            key_2_index[k]=tdd.key_2_index[k]
            
    for k in tdd.index_2_key:
        if isinstance(k,str) and k[0]==x:
            index_2_key[y+k[1:]]=tdd.index_2_key[k]
        else:
            index_2_key[k]=tdd.index_2_key[k]
    tdd.index_set=idx_set
    tdd.key_2_index=key_2_index
    tdd.index_2_key = index_2_key
    
    
def update_tdd(tdd,cir_head,cir_end):
    tdd1 = simulate(cir_head)
    change_index(tdd1,'x','a')
    change_index(tdd1,'y','x')

    tdd2 = simulate(cir_end)
    change_index(tdd2,'y','b')
    change_index(tdd2,'x','y')
    
    tdd_new=cont(tdd,tdd1)
    tdd_new=cont(tdd_new,tdd2)

    change_index(tdd_new,'a','x')
    change_index(tdd_new,'b','y')
    # tdd_new=renormalize(tdd_new)
    return tdd_new

def get_downward_k_level_map(tdd,k,c):
    if k==0:
        if c==0:
            return tdd.node.out_maps[1],tdd.node.out_weight[1]
        else:
            return tdd.node.successor[0].out_maps[1],tdd.node.successor[0].out_weight[1]
    temp_tdd = Slicing2(Slicing2(tdd,tdd.node.key,0),tdd.node.key-1,0)
    return get_downward_k_level_map(temp_tdd,k-1,c)

def get_branch_dd(u,c):
    bran_tdd = TDD(u.successor[c])
    n2 = bran_tdd.node.key+1
    bran_tdd.index_set = []
    bran_tdd.key_2_index ={-1:-1}
    bran_tdd.key_width = {}
    for k in range(n2):
        bran_tdd.index_set.append(Index('y'+str(k),0))
        bran_tdd.key_2_index[k] = 'y'+str(k)
        bran_tdd.key_width[k] = 2
    bran_tdd.index_2_key = {bran_tdd.key_2_index[a]:a for a in bran_tdd.key_2_index} 
    return bran_tdd



def get_meas_prob(node):
    if node.meas_prob!=[]:
        return
    if node.key==-1:
        return
    if node.key==0:
        node.meas_prob = [abs(node.out_weight[0])**2,abs(node.out_weight[1])**2]
        return
    if node.successor[0].meas_prob==[]:
        get_meas_prob(node.successor[0])
    if node.successor[1].meas_prob==[]:
        get_meas_prob(node.successor[1])
    node.meas_prob.append(abs(node.out_weight[0])**2*sum(node.successor[0].meas_prob))
    node.meas_prob.append(abs(node.out_weight[1])**2*sum(node.successor[1].meas_prob))
    node.successor[0].ref+=1
    # if node.successor[1]!=node.successor[0]:
    node.successor[1].ref+=1

    return


def get_ref(node,R=[]):
    if node.key==-1:
        return
    node.ref+=1
    # print('R',R)
    if not node in R:
        node.ref = 1
        R.append(node)
        get_ref(node.successor[0],R)
        get_ref(node.successor[1],R)        
    return


def get_gate_data(cir):
    data = {}
    for d in cir.data:
        a = len(d.qubits)
        if a in data:
            data[a]+=1
        else:
            data[a]=1
    return data

def get_gate_data2(cir):
    data = 0
    for d in cir.data:
        data += len(d.qubits)
    return data

def random_binary_tensor(n: int) -> np.ndarray:
    """
    生成一个形状为 (2, 2, ..., 2)（共 n 个 2）的随机 0/1 数组。
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    shape = (2,) * n
    return np.random.randint(0, 2, size=shape, dtype=np.uint8)


def tensor_to_vector(tensor: np.ndarray) -> np.ndarray:
    """
    把任意形状的 0/1 数组展平成一维向量（按 C-order 行优先）。
    """
    # 也可以用 np.ravel 或 flatten，结果一样
    return tensor.reshape(-1)

def vector_to_tensor(vec: np.ndarray, n: int) -> np.ndarray:
    """
    将长度为 2^n 的 0/1 向量还原成形状为 (2, 2, ..., 2)（共 n 个 2）的数组。
    """
    if vec.ndim != 1:
        raise ValueError("输入必须是 1 维向量")
    if vec.size != (1 << n):          # 1 << n 等价于 2**n
        raise ValueError(f"向量长度 {vec.size} 不等于 2^{n} = {(1 << n)}")
    return vec.reshape((2,) * n)

def get_ramdom_TDD(n,data=np.array([]),var_order_s = []):
    if sum(data.shape)==0:
        data = random_binary_tensor(n)
    # print(data)
    var = []
    var_order = []
    for k in range(n-1,-1,-1):
        var.append(Index('x'+str(k)))
        var_order.append('x'+str(k))
    # print(var_order)
    ts = Tensor(data,var)
    if not var_order_s:
        var_order_s=var_order
    # print(var_order)
    Ini_TDD(var_order_s)
    return ts.tdd(),tensor_to_vector(data)

def get_truth_table(n, tt):
    array = []
    pivot = 1
    for i in range(2 ** n):
        array.append((1 if tt & pivot else 0))
        pivot <<= 1
    return np.array(array).reshape((2,) * n)

def get_TDD_from_truth_table(n, tt,var_order_s = []):
    data = get_truth_table(n, tt)
    var = []
    var_order = []
    for k in range(n-1,-1,-1):
        var.append(Index('x'+str(k)))
        var_order.append('x'+str(k))
    # print(var_order)
    ts = Tensor(data,var)
    if not var_order_s:
        var_order_s=var_order    
    Ini_TDD(var_order_s)
    return ts.tdd(),tensor_to_vector(data)

def swap_two_gate(n,g0,g1):
    "g0在左，g1在右"
    cir_temp = Circuit(n,[])

    # 两门互逆相消
    if g0.q_c==g1.q_c and g0.q_t==g1.q_t:
        return cir_temp,True

        
    #某处控制条件不同:
    com_q_c = [k for k in g0.q_c if k in g1.q_c]
    c = [abs(g0.q_c[k]-g1.q_c[k]) for k in com_q_c]

####-----------处理特殊情况


    # if sum(c)==2 and set(g0.q_c.keys())==set(g1.q_c.keys()) and g0.q_t==g1.q_t: #都是x,某两处控制条件不同
    #     p0 = c.index(1)
    #     p1 = p0+1+c[p0+1:].index(1)
    #     cond0={com_q_c[p0]:1}
    #     cond1={com_q_c[p1]:1}
    #     for k in g0.q_c:
    #         if k!=p0 and k!=p1:
    #             cond0[k] = g0.q_c[k]
    #             cond1[k] = g0.q_c[k]
    #     g00 = Gate('x',cond0,g0.q_t,g0.para)
    #     g11 = Gate('x',cond1,g1.q_t,g1.para)
    #     cir_temp.data.append(g00)
    #     cir_temp.data.append(g11)
    #     if g0.q_c[com_q_c[p0]]==g0.q_c[com_q_c[p1]]:
    #         g = Gate('x',{},g1.q_t,g1.para)
    #         cir_temp.data.append(g)
    #     return cir_temp,True

                


####-----------处理一般情况
    
    #仅控制比特相交或完全不相交
    if g0.q_t!=g1.q_t and not g0.q_t in g1.q_c and not g1.q_t in g0.q_c:
        cir_temp.data = [g1,g0]
        return cir_temp,True


    # print('---------------')
    # print(g0,g1)
    if sum(c)==1 and g0.q_t==g1.q_t:
        res0 = [k for k in g0.q_c if not k in com_q_c]
        res1 = [k for k in g1.q_c if not k in com_q_c]
        if len(res0)==0 and len(res1)==1:
            com_cond = {}
            new_cond = {}
            for k in g1.q_c:
                if k in g0.q_c and g0.q_c[k]==g1.q_c[k]:
                    com_cond[k]=g1.q_c[k]
                    new_cond[k]=g1.q_c[k]
                if not k in g0.q_c:
                    new_cond[k]=1-g1.q_c[k]
                else:
                    new_cond[k]=g1.q_c[k]
            g00 = Gate(g0.name,com_cond,g0.q_t)
            g11 = Gate(g0.name,new_cond,g0.q_t)
            cir_temp.data.append(g00)
            cir_temp.data.append(g11)
            return cir_temp,True
        if len(res0)==1 and len(res1)==0:
            com_cond = {}
            new_cond = {}
            for k in g0.q_c:
                if k in g1.q_c and g0.q_c[k]==g1.q_c[k]:
                    com_cond[k]=g0.q_c[k]
                    new_cond[k]=g0.q_c[k]
                if not k in g1.q_c:
                    new_cond[k]=1-g0.q_c[k]
                else:
                    new_cond[k]=g0.q_c[k]
            g00 = Gate(g0.name,new_cond,g0.q_t)
            g11 = Gate(g0.name,com_cond,g0.q_t)
            cir_temp.data.append(g00)
            cir_temp.data.append(g11)
            return cir_temp,True
    
    # print('------==-------')
    if sum(c)==1 and set(g0.q_c.keys())==set(g1.q_c.keys()) and g0.q_t==g1.q_t:#比特相同，某一处控制条件不同
        q = [k for k in g0.q_c]
        q.remove(com_q_c[c.index(1)])
        cond = {k:g0.q_c[k] for k in q}
        g = Gate(g0.name,cond,g0.q_t,g0.para)
        cir_temp.data.append(g)
        return cir_temp,True

    
    if sum(c)>0:#某处控制条件不同         
        cir_temp.data = [g1,g0]
        return cir_temp,True
        
    if g0.q_t==g1.q_t:
        if len(g0.q_c)==len(g1.q_c)+1: #控制条件相同但差一个比特
            q = [k for k in g0.q_c]
            cond={}
            for k in q:
                if k in g1.q_c:
                    cond[k] = g0.q_c[k]
                else:
                    cond[k] = 1-g0.q_c[k]
            g = Gate(g0.name,cond,g0.q_t,g0.para)
            cir_temp.data.append(g)
            return cir_temp,True
        if len(g1.q_c)==len(g0.q_c)+1: #控制条件相同但差一个比特
            q = [k for k in g1.q_c]
            cond={}
            for k in q:
                if k in g0.q_c:
                    cond[k] = g1.q_c[k]
                else:
                    cond[k] = 1-g1.q_c[k]
            g = Gate(g1.name,cond,g1.q_t,g1.para)
            cir_temp.data.append(g)
            return cir_temp,True
            

        
    if g0.q_t == g1.q_t:
        cir_temp.data = [g1,g0]
        return cir_temp,True


    if g0.name=='x' and g0.q_t in g1.q_c and set(g0.q_c.keys()).issubset(g1.q_c.keys()):
        cond = copy.copy(g1.q_c)
        cond[g0.q_t] = 1-cond[g0.q_t]
        g = Gate(g1.name,cond,g1.q_t,g1.para)
        cir_temp.data = [g,g0]
        return cir_temp,True
    if g1.name=='x' and g1.q_t in g0.q_c and set(g1.q_c.keys()).issubset(g0.q_c.keys()):
        cond = copy.copy(g0.q_c)
        cond[g1.q_t] = 1-cond[g1.q_t]
        g = Gate(g0.name,cond,g0.q_t,g0.para)
        cir_temp.data = [g1,g]
        return cir_temp,True  
    
    cir_temp.data = [g0,g1]
    return cir_temp,False

def swap_a_gate_to_head(cir_l,g):
    n=cir_l.num_qubits
    cir_new = Circuit(n,[])
    if len(cir_l.data)==0:
        cir_new.data.append(g)
        return cir_new,True
    for k in range(len(cir_l.data)-1,-1,-1):
        cir_t,res = swap_two_gate(n,cir_l.data[k],g)
        if not res:
            cir_new.data=cir_l.data[:k]+cir_t.data+cir_new.data
            return cir_new,res
        if k!=0:
            if len(cir_t.data)==0:
                cir_new.data = cir_l.data[:k]+cir_new.data
                break
            g = cir_t.data.pop(0)
        cir_new.data = cir_t.data+cir_new.data       
    return cir_new,True


def mcx_opt_pass(cir):
    n2 = cir.num_qubits
    cir_l = Circuit(n2,[])
    for g in cir.data:
        cir_t,res = swap_a_gate_to_head(cir_l,g)
        cir_l = cir_t
    return cir_l
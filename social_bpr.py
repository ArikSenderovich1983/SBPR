
import random
import numpy as np
random.seed(30)
import os
import networkx as nx
import pymzn
import matplotlib.pyplot as plt
from pymzn import Solver
from pymzn import Status
import time as tm
import pandas as pd
class MySolver(Solver):
    def __init__(self):
        super().__init__(solver_id='gurobi')
def random_edge(graph, del_orig=True):
    '''
    Create a new random edge and delete one of its current edge if del_orig is True.
    :param graph: networkx graph
    :param del_orig: bool
    :return: networkx graph
    '''
    edges = list(graph.edges)
    nonedges = list(nx.non_edges(graph))

    # random edge choice
    chosen_edge = random.choice(edges)
    chosen_nonedge = random.choice([x for x in nonedges if chosen_edge[0] == x[0]])

    if del_orig:
        # delete chosen edge
        graph.remove_edge(chosen_edge[0], chosen_edge[1])
    # add new edge
    graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])

    return graph
def gen_SBPR_automation(n_A,n_R,n_S):
    # we wish to generate SBPR instances

    #Universe of activities:
    A= n_A
    u_activities = ['a'+str(i) for i in range(1,A+1)]


    # In[3]:


    #Resources
    R = n_R
    u_resources = ['r'+str(i) for i in range(1,R+1)]
    print(u_resources)
    #Last resource is always going to be the machine (we add later)


    # In[4]:


    #generate skills graph (directed)
    S = n_S
    G = nx.generators.gn_graph(n=S, seed=5)
    #DAG = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
    if nx.is_directed_acyclic_graph(G)==False:
        print('Broken')
        return {}

    # In[5]:


    nx.draw(G)
    #plt.draw()

    # In[6]:


    #generating skills.

    skills = [i+1 for i in list(G.nodes)]
    print(skills)


    # In[7]:


    #matching skills to activities.
    s_a = []
    for a in u_activities:
        s_a.append(random.choice(skills))

    print(s_a)


    # In[8]:


    Sigma = np.random.randint(2, size=(R,S))
    print(Sigma)


    # In[9]:


    Sigma = np.append(Sigma , [np.zeros(S, dtype = np.int32)], axis = 0)
    print(Sigma)


    # In[10]:


    u_resources.append('m')
    R = len(u_resources)
    print(R)


    # In[11]:


    prec = nx.adjacency_matrix(G).todense()
    print(prec)


    # In[12]:


    weights = np.ones((R-1,A),dtype = np.int32)
    weights = np.append(weights , [np.zeros(A, dtype = np.int32)], axis = 0)
    print("FTE weights:")
    print(weights)


    # In[13]:


    social_costs = np.zeros((R-1,A),dtype = np.int32)
    social_costs = np.append(social_costs , [np.ones(A, dtype = np.int32)], axis = 0)
    print('Social cost:')

    print(social_costs)


    # In[14]:


    social_budget = []
    for r in range(0,R-1):
        social_budget.append(A)
    #automation budget:
    social_budget.append(random.randint(0, A))
    print('Social budget:')
    print(social_budget)


    # In[15]:


    #Learning:
    #idea: orders of mag.
    learn_possibility = [0,1,10,100,1000,10000,100000]
    learning = np.zeros((R,S),dtype = np.int32)
    for i in range(R):
        for j in range(S):
            learning[i,j] = random.choice(learn_possibility)
    print('Learning cost:')

    print(learning)



    instance_data = {'A': A,
                                 'R': R,#sum_proc,#cur_horizon,
                                 'S': S,
                                 'Sigma': list(Sigma.tolist()),
                                 'S_a': s_a,
                                 'prec': list(prec.tolist()),
                                 'weights': list(weights.tolist()),
                                 'social_cost': list(social_costs.tolist()),
                                 'social_budget': social_budget,
                                 'learning': list(learning.tolist())
                                 }

    #print(instance_data)
    return instance_data
def gen_SBPR_Completely_Random(A, R, S, verbose, density, unsocial, util):
    # we wish to generate SBPR instances
    #Universe of activities:

    u_activities = ['a'+str(i) for i in range(1,A+1)]

    #Resources

    u_resources = ['r'+str(i) for i in range(1,R+1)]

    #generate skills graph (directed)
    G = random_dag(S, edges = density*S*S)
    #G = nx.generators.gn_graph(n=S, seed=5)

    if nx.is_directed_acyclic_graph(G)==False:
        print('Not DAG')
        return ValueError

    #nx.draw(G)

    nx.drawing.draw_random(G)




    #generating skills.

    skills = [i+1 for i in list(G.nodes)]


    #matching skills to activities.
    s_a = []
    for a in u_activities:
        s_a.append(random.choice(skills))



    Sigma = np.random.randint(2, size=(R,S))

    prec = nx.adjacency_matrix(G).todense()

    M = random.sample(range(0,R), k=max(int(unsocial*R),1))
    print(M)
    # In[12]:

    #weight_possibility = [1,10,100, 1000, 10000]
    #weight_possibility = [1,2,3,4]
    weight_possibility = range(10, 26)
    weights = np.ones((R,A),dtype = np.int32)

    for i in range(R):
        for j in range(A):
            #if i in M:
                #we want low cost
                #weights[i, j] = 0
            #else:
            weights[i,j] = random.choice(weight_possibility)

    #weights = np.append(weights , [np.zeros(A, dtype = np.int32)], axis = 0)



    # Learning:
    # idea: orders of mag.
    #machines also learn.
    learn_possibility = range(5,26)# [1, 10, 100, 1000, 10000]
    learning = np.zeros((R, S), dtype=np.int32)
    for i in range(R):
        for j in range(S):
            learning[i, j] = random.choice(learn_possibility)

    social_costs = np.zeros((R,A),dtype = np.int32)

    #social_possibility = [0, 1, 10]
    #social_possibility = range(0,2)
    social_possibility = range(0,26)
    for i in range(R):
        for j in range(A):
            #if i in M:
                social_costs[i, j] = random.choice(social_possibility)
            #else:

                #social_costs[i,j] = 0


    social_budget = []

    #K = range(1,max(social_possibility)*A)

    #first select the unsocial roles out of R:


    K = range(1, int(2*A/R-1))# int(2*A/R-1))


    for i in range(0,R):
        sum_a = 0
        for j in range(A):
            sum_a+=social_costs[i,j]
        #print(sum_a)
        social_budget.append(int(util / R *sum_a))


    #for r in range(0,R):
        #if r in M:
            #social_budget.append(random.choice(K)) #1)
    #        social_budget.append(int(np.ceil(A/R)))
        #else:
        #    social_budget.append(A) #random.choice(K))


        #social_budget.append(random.choice(K))

    print(social_costs)
    print(social_budget)

    instance_data = {'A': A,
                                 'R': R,#sum_proc,#cur_horizon,
                                 'S': S,
                                 'Sigma': list(Sigma.tolist()),
                                 'S_a': s_a,
                                 'prec': list(prec.tolist()),
                                 'weights': list(weights.tolist()),
                                 'social_cost': list(social_costs.tolist()),
                                 'social_budget': social_budget,
                                 'learning': list(learning.tolist())
                                #'quality':list(quality.tolist()),
                                #'quality_lower': quality_lower
                                 }


    if verbose:
        print(instance_data)
        plt.show()


    return instance_data

def gen_GAP_Completely_Random(A, R, verbose):
    # we wish to generate SBPR instances
    #Universe of activities:

    u_activities = ['a'+str(i) for i in range(1,A+1)]



    weight_possibility = list(range(10,26))
    #weight_possibility = [1,2,3,4]

    weights = np.ones((R,A),dtype = np.int32)

    for i in range(R):
        for j in range(A):
            #if i in M:
                #we want low cost
                #weights[i, j] = 0
            #else:
            weights[i,j] = random.choice(weight_possibility)

    #weights = np.append(weights , [np.zeros(A, dtype = np.int32)], axis = 0)




    social_costs = np.zeros((R,A),dtype = np.int32)

    #social_possibility = [0, 1, 10]
    social_possibility = range(5,26)
    for i in range(R):
        for j in range(A):
            #if i in M:
                social_costs[i, j] = random.choice(social_possibility)
            #else:

                #social_costs[i,j] = 0


    social_budget = []

    #K = range(1,max(social_possibility)*A)

    #first select the unsocial roles out of R:


    K = range(1, int(2*A/R-1))# int(2*A/R-1))

    for i in range(0,R):
        sum_a = 0
        for j in range(A):
            sum_a+=social_costs[i,j]
        print(sum_a)
        social_budget.append(int(0.9/R *sum_a))


        #social_budget.append(random.choice(K))

    print(social_costs)
    print(social_budget)

    instance_data = {'A': A,
                                 'R': R,#sum_proc,#cur_horizon,
                                 'weights': list(weights.tolist()),
                                 'social_cost': list(social_costs.tolist()),
                                 'social_budget': social_budget
                                #'quality':list(quality.tolist()),
                                #'quality_lower': quality_lower
                                 }


    if verbose:
        print(instance_data)
        plt.show()


    return instance_data

def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = random.randint(0,nodes-1)
        b=a
        while b==a:
            b = random.randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)
    return G

#feasibility test
#scenarios = [(10,5,5)]#, (100,5,5), (100, 50, 5), (100, 50, 50), (10, 10, 10), (10, 10, 500)]
import csv

def run_experiment():

    def write_csv(data):
        with open('intermediate_results.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data)



    #basic process, (10,5,5)
    #max process (10000,200,50)

    activities_set = [100, 200, 300]
    ratios = [5,10]

    #M=45

    skills_set = [50,100]
    M=0.95
    utils = [0.5, 0.3, 0.1]


    #for density in densities:


    densities = range(1, 10, 4)
    densities = [d/10 for d in densities]

    #densities = [0.5]

    scenarios = [(a,int(a/r),s,d, u) for a in activities_set for r in ratios for s in skills_set
                 for d in densities for u in utils]
    string_scenarios = []
    for s in scenarios:
        string_scenarios.append(str(s[0]) + '_'+str(s[1]) + '_'+str(s[2])+'_'+str(s[3])+'_'+str(s[4]))
    runtimes_first = []
    runtimes_last = []
    runtimes_proof = []
    attempts =[]
    #machines out of total
    objs = []
    for j,s in enumerate(scenarios):
        attempts.append(0)
        infeasible_flag = True
        while infeasible_flag==True:
            attempts[len(attempts)-1]+=1
            print('Current scenario running: '+string_scenarios[j])

            n_A, n_R, n_S, density, util= s

            try:
                instance_data = gen_SBPR_Completely_Random(n_A,n_R,n_S,False, density, M, util)#M/n_R)#(n_R-1)/n_R) #0.5)
                #instance_data2 = gen_GAP_Completely_Random(n_A,n_R,verbose=True)
            except ValueError:
                print("Error")
                continue
            #instance_data = gen_SBPR(n_A,n_R,n_S)
            print(os.path.join("data_files","SBPR_"+str(s)+".dzn"))
            pymzn.dict2dzn(instance_data, fout=os.path.join("data_files","SBPR_"+str(s)+".dzn"))
            #pymzn.dict2dzn(instance_data2, fout="arik_Gap.dzn")

            mysolver = MySolver()
            print('start solving...')
            mzn_sol = pymzn.minizinc("SBPR_problem.mzn", data= instance_data, solver=mysolver, all_solutions = True)
            #mzn_sol = pymzn.minizinc("GAP_v1.mzn", data= instance_data2, solver=mysolver, all_solutions = True)

            #mzn_sol = pymzn.minizinc("arik-problem-v2.mzn", data= instance_data,
            #solver =pymzn.Gecode(solver_id = 'gecode'))
            #         solver='chuffed')# solver = pymzn.Chuffed(solver_id='chuffed'))

            print(mzn_sol)
            if mzn_sol.status== Status.COMPLETE:
                objs.append((int(mzn_sol._solns[0]['Z']), int(mzn_sol._solns[0]['L'])))
                #objs.append(mzn_sol._solns[0]['Z'])

                split_obj = mzn_sol.log.split('objective=')[1:]
                runtimes_first.append(float(mzn_sol.log.split('objective=')[1:][0].split('solveTime=')[1].split('\n')[0]))
                runtimes_last.append(float(mzn_sol.log.split('objective=')[1:][-2].split('solveTime=')[1].split('\n')[0]))
                runtimes_proof.append(float(mzn_sol.log.split('objective=')[1:][-1].split('solveTime=')[1].split('\n')[0]))
                write_csv([s, float(mzn_sol.log.split('objective=')[1:][-1].split('solveTime=')[1].split('\n')[0])])
                print(runtimes_first[len(runtimes_first)-1])
                print(runtimes_proof[len(runtimes_proof)-1])
                #print('Time to optimal:')
                #print(tm.time() - start_time)
                #runtimes.append(tm.time() - start_time)
                infeasible_flag=False
            else:
                print('Infeasible for scenario '+str(s))
                print('Retrying...')

    dict_ = {'scenarios': string_scenarios,'objectives':objs, 'runtimes_first':runtimes_first,
         'runtimes_last':runtimes_last, 'runrtimes_proof':runtimes_proof,
         'attempts':attempts}

    res_df = pd.DataFrame(dict_)
    print(res_df.head(10))
    res_df.to_csv(os.path.join('results','results.csv'))


run_experiment()

#res_df=pd.read_csv("intermediate_results.csv")
#print(res_df.head(5))
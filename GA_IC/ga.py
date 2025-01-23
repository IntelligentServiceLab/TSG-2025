
#%% 读取IEEE33配电网数据
# bigM
bigM = 100
basekV = 12.66# 电压基准值为12.66kV
baseMVA = 10 # 功率基准值为10MVA
baseI = 789.89
# 节点数据
T_set = np.arange(24) # //时间
dT = 1
B_num = 33
E_num = 37
B_set = np.arange(33)
"""
//不同类型节点的均值和标准差
"""
f0 = open("支路连接情况.txt")
t=33
PQ = np.zeros((t, 2)) # //用于存储与每个节点相关的功率信息，其中每一行代表一个节点，第一列可能存储实功率（P），第二列存储虚功率（Q）
A0 = np.zeros((t, t), int) # //用于存储节点之间的连接关系或者网络的拓扑结构
SB = 10  # 功率基准
#fault_branch=[9,25]
fault_branch=[9]
branch_num = 33
branch_node = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
                   [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21], [2, 22],
                   [22, 23], [23, 24],
                   [5, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [20, 7], [8, 14],
                   [11, 21], [17, 32], [24, 28]]
population_start = [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0] #用于计算开关次数
population_start_change =[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] #用于编码转换的156151
population_ic_start = []
M = [[1,2,3,4,5,6,32,19,18,17],[33,13,12,11],[20,34,10,9,8,7],[14,15,16,35,30,29,28],[24,25,26,27,36,23,22,21]] #未发生故障时配电网环路对应的开关序号
#N = [[1,2,3,4,5,6,32,19,18,17],[33,13,12,11],[0,0,0,9],[14,15,16,35,31,30,29,28],[0,25]] #删除故障所在行后的M
#N = [[1,2,3,4,5,6,32,19,18,17],[33,13,12,11],[9],[14,15,16,35,31,30,29,28],[25]]
N = [[1,2,3,4,5,6,32,19,18,17],[33,13,12,11],[9],[14,15,16,35,31,30,29,28],[24,25,26,27,36,23,22,21]]
#population_max = [10,4,1,8,1]
population_max = [10,4,1,8,8]
population_min = [1,1,1,1,1] #种群的最小值
task_number= 5 #参与运输的行数
population_size = 20
iteration_number= 20
#population_max = [10,4,6,8,8]  未考虑故障时各维度的最大值
bounds = np.vstack((population_max, population_min))
C1=0.4
C2=0.2
C3=0.4#按环状编码加划分矩阵的方法是能够全部恢复的
class GA():
    """粒子群算法"""

    def __init__(self, population_x, task_number, population_size, iteration_number, bounds,fault_branch):

        self.cp = 0.5  # cp为交叉率
        self.mp = 0.88  # mp为变异率
        self.bounds = bounds  # 位置的边界
        self.population_x = population_x  # 初始种群
        self.task_number = task_number  # 任务数
        self.population_size = population_size  # 种群规模(粒子数量)
        self.iteration_number = iteration_number  # 迭代次数
        self.fault_brach = fault_branch  # 故障支路
    def initialization(self, populathion_x):
        population = self.population_x  # 种群
        return population

    # 编码转换
    def population_change(self, population_x):
        new_popolation = copy.deepcopy(population_start_change)
        for i in range(task_number):
            j = int(population_x[i] - 1)
            x = N[i][j]
            new_popolation[x] = 0
        return new_popolation

    def Selection(self, population,population_island):

        new_population = []
        new_population_island=[]
        tournament_size = 2  # 锦标赛规模

        # 锦标赛
        for i in range(0, self.population_size):
            temp = copy.deepcopy(population)  # 临时列表，供锦标赛抽取
            temp_island=copy.deepcopy(population_island)
            a=random.randint(0, len(temp)-1)
            b=random.randint(0, len(temp)-1)
            competitor_a = copy.deepcopy(population[a])  # 随机抽取选手a
            competitor_a_island = copy.deepcopy(population_island[a])
            competitor_b = copy.deepcopy(population[b])  # 随机抽取选手b
            competitor_b_island = copy.deepcopy(population_island[b])

            # 若a支配b
            if self.total_fitness(fitness_loss(competitor_a_island), self.fitness_switch(competitor_a_island),self.fitness_node(competitor_a_island))<self.total_fitness(fitness_loss(competitor_b_island), self.fitness_switch(competitor_b_island),self.fitness_node(competitor_b_island)):
                new_population.append(competitor_a)
                new_population_island.append(competitor_a_island)

            # 若b支配a
            elif self.total_fitness(fitness_loss(competitor_a_island), self.fitness_switch(competitor_a_island),self.fitness_node(competitor_a_island))>self.total_fitness(fitness_loss(competitor_b_island), self.fitness_switch(competitor_b_island),self.fitness_node(competitor_b_island)):
                new_population.append(competitor_b)
                new_population_island.append(competitor_b_island)
            # 若互相不支配
            else:
                new_population.append(population[i])
                new_population_island.append(population_island[i])
        return new_population,new_population_island

    def Crossover(self, population,population_island):
        """交叉操作"""

        cp = self.cp  # 交叉概率
        new_population = []  # 初始化交叉完毕的种群
        new_population_island =np.full((population_size,E_num),0)
        crossover_population = []  # 初始化需要交叉的种群
        # 根据交叉概率选出需要交叉的个体
        for c in population:
            r = random.random()
            if r <= cp:
                crossover_population.append(c)
            else:
                new_population.append(c)

        # 需保证交叉的个体是偶数,若不是偶数，则删掉需交叉列表的最后一个元素
        if len(crossover_population) % 2 != 0:
            new_population.append(crossover_population[len(crossover_population) - 1])
            del crossover_population[len(crossover_population) - 1]

        # crossover——单点交叉
        for i in range(0, len(crossover_population), 2):

            i_solution = crossover_population[i]
            j_solution = crossover_population[i + 1]
            crossover_position = random.randint(1, self.task_number - 2)  # 随机生成一个交叉位
            left_i = copy.deepcopy(i_solution[0:crossover_position])
            right_i = copy.deepcopy(i_solution[crossover_position:self.task_number])
            left_j = copy.deepcopy(j_solution[0:crossover_position])
            right_j = copy.deepcopy(j_solution[crossover_position:self.task_number])
            # 生成新个体a
            new_i = np.concatenate((left_i,right_j))
            new_j = np.concatenate((left_j,right_i))
            new_population.append(new_i)
            new_population.append(new_j)

            if (i + 1) == (len(crossover_population) - 1):
                break
        for i in range(population_size):
            new_population_island[i]=find_island(self.population_change(new_population[i]))
        return new_population,new_population_island

    def Mutation(self, population,population_island):
        """变异操作"""
        mp = self.mp  # 变异率
        new_population = [] # 初始化变异后的种群
        new_population_island=np.full((population_size,E_num),0)
        for c in population:
            r = random.random()
            if r <= mp:
                # mutation——随机选择某个体的一个任务（位置），从对应候选服务集中随机选择某服务替换
                mutation_position = random.randint(0, self.task_number - 1)  # 变异位置
                mutation_pop=random.randint(population_min[mutation_position], population_max[mutation_position])
                c[mutation_position] = copy.deepcopy(mutation_pop)
                new_population.append(c)
            else:
                new_population.append(c)

        for i in range(population_size):
            new_population_island[i]=find_island(self.population_change(new_population[i]))
        return new_population,new_population_island


    def save_gbest(self, population,population_island):
        """更新种群历史最优"""

        min = 0
        for i in range(0, self.population_size):
            if self.total_fitness(fitness_loss(population_island[min]),
                                  self.fitness_switch(population_island[min]),self.fitness_node(population_island[min])) > self.total_fitness(
                    fitness_loss(population_island[i]),
                    self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])) :
                min = i

        return population[min],min

    def sigmoid(self,V):
        return 1 / (1 + np.exp(-V))
    def min_max_turnone(self,fitness, min_value, max_value):
        normalized_value = (fitness - min_value) / (max_value - min_value)
        return normalized_value

    def total_fitness(self, fit_loss, fit_switch, fit_node):
        total_fit = 0
        total_fit = C1 * self.min_max_turnone(fit_loss, min_max_fit_loss[0],
                                              min_max_fit_loss[1]) - C2 * self.min_max_turnone(fit_switch,
                                                                                               min_max_fit_switch[0],
                                                                                               min_max_fit_switch[
                                                                                                   1]) - C3 * self.min_max_turnone(
            fit_node, min_max_fit_node[0], min_max_fit_node[1])
        return total_fit

    def find_teacher(self, population,population_island):
        """找到种群中的老师(Pareto解集)"""

        min = 0
        for i in range(0, self.population_size):
            if self.total_fitness(fitness_loss(population_island[min]),self.fitness_switch(population_island[min]),self.fitness_node(population_island[min]))>self.total_fitness(fitness_loss(population_island[i]),self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])):
                min = i
        return population[min],min

    def find_max(self, population,population_island):
        """找到种群中的老师(Pareto解集)"""

        max = 0
        for i in range(0, self.population_size):
            if self.total_fitness(fitness_loss(population_island[max]),self.fitness_switch(population_island[max]),self.fitness_node(population_island[max]))<self.total_fitness(fitness_loss(population_island[i]),self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])):
                max = i
        return population[max]

    """
    DG利用率
    """
    def fitness_switch(self, solution):
        all_nodes = list(range(33))
        G = create_adjacency_matrix(branch_node, solution, branch_num)
        connected_nodes = list(nx.node_connected_component(nx.from_numpy_array(G), 0))
        remaining_nodes = list([node for node in all_nodes if node not in connected_nodes])
        sum_load = 0
        DG_tatol_load = 0
        num = 0
        for i in DG_load:
            DG_tatol_load = DG_tatol_load + i
        for j in remaining_nodes:
            sum_load = sum_load + load[j]
        num = sum_load / DG_tatol_load
        return num

    def find_connected_components(self,graph):
        # 获取图的节点数
        num_nodes = graph.shape[0]

        # 初始化一个布尔数组，标记每个节点是否已被访问
        visited = [False] * num_nodes

        # 用于存储所有连通分量
        connected_components = []

        # 定义深度优先搜索 (DFS) 函数
        def dfs(node, component):
            visited[node] = True  # 标记当前节点为已访问
            component.append(node)  # 将当前节点加入到连通分量中

            # 遍历所有邻接节点
            for neighbor in range(num_nodes):
                if graph[node][neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor, component)

        # 遍历每个节点
        for node in range(num_nodes):
            if not visited[node]:  # 如果当前节点没有被访问过
                component = []  # 当前连通分量
                dfs(node, component)  # 从当前节点开始DFS
                connected_components.append(component)  # 将当前连通分量加入到结果列表

        return connected_components

    # 恢复率
    def fitness_node(self, solution):
        loadd = 0
        G = create_adjacency_matrix(branch_node, solution, branch_num)
        component = self.find_connected_components(G)  # 找到连通分量
        components_num = len(component)  # 有几个连通分量
        for i in range(components_num):
            z = 0
            for j in component[i]:
                if j in PS:  # 连接了电源
                    z = 1
                    continue
            if z == 1:
                for l in component[i]:
                    loadd += load[l]
        num = loadd / total_load
        return num

    def update(self, old_group,old_group_island, new_group,new_group_island):


        updated_group = []
        updated_group_island = []
        for i in range(self.population_size):
            # 如果新解支配旧解
            if self.total_fitness(fitness_loss(new_group_island[i]), self.fitness_switch(new_group_island[i]),self.fitness_node(new_group_island[i])) < self.total_fitness(
                    fitness_loss(old_group_island[i]), self.fitness_switch(old_group_island[i]),self.fitness_node(old_group_island[i])):
                updated_group.append(new_group[i])
                updated_group_island.append(new_group_island[i])
                continue
            else:
                updated_group.append(old_group[i])
                updated_group_island.append(old_group_island[i])
        return updated_group, updated_group_island

    def run_GA(self):
        #初始化种群和参数
        # 初始化种群和参数
        # start_time = time.time()
        record=[]
        gbestNum=0
        old_pop = self.initialization(self.population_x)
        old_pop_island=np.full((population_size,E_num),0)
        for i in range(population_size):
            old_pop_island[i]=find_island(self.population_change(old_pop[i]))
        gbest,gbestNum= self.find_teacher(old_pop,old_pop_island)
        ig=0
        sum_sum=[]
        for iteration in range(self.iteration_number):

            record.append(fitness_loss(old_pop_island[gbestNum]) * 10000)
            sum1=0
            numnum=0
            for i in range(population_size):
                    sum1=sum1+fitness_loss(old_pop_island[i])*10000
                    numnum+=1
            sum1=float(sum1/numnum)
            sum_sum.append(sum1)

            new_pop,new_pop_island = self.Selection(old_pop,old_pop_island)
            old_pop, old_pop_island = self.update(old_pop, old_pop_island, new_pop, new_pop_island)
            new_pop,new_pop_island = self.Crossover(old_pop,old_pop_island)
            old_pop, old_pop_island = self.update(old_pop, old_pop_island, new_pop, new_pop_island)
            new_pop,new_pop_island = self.Mutation(old_pop,old_pop_island)
            old_pop,old_pop_island=self.update(old_pop,old_pop_island,new_pop,new_pop_island)

            gbest,gbestNum = self.save_gbest(old_pop,old_pop_island)
        final_Solution = old_pop_island[gbestNum]
        print(final_Solution)
        island_rate = self.fitness_switch(final_Solution)
        # 输出教师作为最终结果
        totalss = 0
        for l in sum_sum:
            totalss = totalss + l
        totals = totalss/self.iteration_number
        print(totals)#平均
        return island_rate,gbest,record,old_pop,sum_sum

start_time = time.time()
result = GA(x_solutions.create_population(population_size,N,fault_branch),task_number,population_size,iteration_number,bounds,fault_branch)
island_rate,gbest,record,end_pop,sum_sum = result.run_GA()
end_time = time.time()
print("GA_IC")
print('结束')





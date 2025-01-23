
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
class PSO():
    """粒子群算法"""

    def __init__(self, population_x, task_number, population_size, iteration_number, bounds,fault_branch):

        self.w = 0.7  # w为惯性因子
        self.c1 = 1.5
        self.c2 = 1.5  # c1, c2为学习因子，一般取1.5
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

    def initialization_V(self, Vmin, Vmax):
        population_V = []  # 速度
        for i in range(0, self.population_size):
            temp = np.zeros(5)

            for j in range(0, self.task_number):
                temp[j] = random.uniform(Vmin[j], Vmax[j])

            population_V.append(temp)

        return population_V

    def get_Vmax(self, bounds):
        """获取速度的上下界"""
        Vmax = []  # 每个任务的速度上界
        # 速度的上界
        for i in range(self.task_number):

            temp = 1 * (bounds[0][i])
            Vmax.append(temp)

        return Vmax

    def get_Vmin(self, bounds):
        """获取速度的上下界"""
        Vmin = []  # 每个任务的速度下界
        for i in range(self.task_number):

            temp = (-1) * (bounds[0][i])
            Vmin.append(temp)
        return Vmin

    def min_max_turnone(self,fitness, min_value, max_value):
        normalized_value = (fitness - min_value) / (max_value - min_value)
        return normalized_value
    def total_fitness(self,fit_loss,fit_switch,fit_node):
        total_fit=0
        total_fit=C1*self.min_max_turnone(fit_loss,min_max_fit_loss[0],min_max_fit_loss[1])-C2*self.min_max_turnone(fit_switch,min_max_fit_switch[0],min_max_fit_switch[1])-C3*self.min_max_turnone(fit_node,min_max_fit_node[0],min_max_fit_node[1])
        return total_fit

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
    def update_X(self, pop_X, pop_V,population_island):
        """更新位置pop_X是一个种群，相当于population_x"""
        new_pop_X = []  # 种群更新后的位置
        new_pop_island=[]
        for i in range(0, self.population_size):
            y=[]
            new_X = np.zeros(5)
            #print(i)
            #print("ppp=",pop_V[i])
            for j in range(0, self.task_number):
                new_X[j] = pop_X[i][j] + pop_V[i][j]

                if new_X[j]>population_max[j]:
                    new_X[j] = population_max[j]#若超过上界则直接设置为最大标号
                elif new_X[j]<population_min[j]:
                    new_X[j]=population_min[j]#若小于下界则直接设置为最小标号
                else:
                    INT=int(new_X[j]/1)
                    yy=new_X[j]%1
                    if yy>=0.5:
                        new_X[j]=INT+1
                    else:
                        new_X[j]=INT
            new_pop_X.append(new_X)
        new_pop_X,new_pop_island=copy.deepcopy(self.update(pop_X,new_pop_X,population_island))

        return new_pop_X,new_pop_island

    def update_V(self, pop_X, pop_V, pbest, gbest, Vmin, Vmax):
        """更新速度"""
        newnew_pop_V=[]
        for i in range(0, self.population_size):
            new_pop_V = []

            for j in range(0, self.task_number):
                r1 = random.random()

                r2 = random.random()
                speed = self.w * pop_V[i][j] + self.c1 * r1 * (pbest[i][j] - pop_X[i][j]) + self.c2 * r2 * (
                        gbest[j] - pop_X[i][j])
                # 判断是否越上界
                if speed > Vmax[j]:
                    speed = Vmax[j]

                # 判断是否越下界
                if speed < Vmin[j]:
                    speed = Vmin[j]

                new_pop_V.append(speed)
            newnew_pop_V.append(new_pop_V)
        return newnew_pop_V

    def save_pbest(self, pbest,pbest_island, pop_X,pop_X_island):
        """更新个体历史最优"""
        updated_pbest = []
        updated_pbest_island = []

        for i in range(self.population_size):
            # 如果新解支配旧解
            if self.total_fitness(fitness_loss(pop_X_island[i]),self.fitness_switch(pop_X_island[i]),self.fitness_node(pop_X_island[i]))<self.total_fitness(fitness_loss(pbest_island[i]),self.fitness_switch(pbest_island[i]),self.fitness_node(pbest_island[i])) :
                updated_pbest.append(pop_X[i])
                updated_pbest_island.append(pop_X_island[i])
                continue
            # if (teacher == pbest[i]).all():
            #     updated_pbest.append(pbest[i])
            #     updated_pbest_island.append(pbest_island[i])
            #     continue
            # if not satisfy_conditions.is_n_equal_to_m_plus_one(pop_X[i]) :
            #     y=base_on_tulun.recovery_x(pop_X[i])
            #     updated_pbest.append(y)
            #     continue"""环状编码后不存在还有拓扑排序问题"""
            else:
                # y = base_on_tulun.recovery_x(solution)
                updated_pbest.append(pbest[i])
                updated_pbest_island.append(pbest_island[i])

        return updated_pbest,updated_pbest_island

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
    def find_teacher(self, population,population_island):
        """找到种群中的老师(Pareto解集)"""
        min = 0
        for i in range(0, self.population_size):
            if self.total_fitness(fitness_loss(population_island[min]),
                                  self.fitness_switch(population_island[min]),self.fitness_node(population_island[min])) > self.total_fitness(
                fitness_loss(population_island[i]), self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])) :
                min = i
        return population[min], min

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

    def update(self, old_group, new_group,population_island):
        """这个函数用来更新种群:若新解支配旧解，则替换;否则保留"""
        final_population_island = np.full((population_size, E_num), 0)
        new_population_island = np.full((population_size, E_num), 0)
        for i in range(self.population_size):
            new_population_island[i] = find_island(self.population_change(new_group[i]))
        updated_group=[]
        for i in range(self.population_size):
            """ 如果新解的总适应度低于旧解 """
            if self.total_fitness(fitness_loss(new_population_island[i]),
                                  self.fitness_switch(new_population_island[i]),self.fitness_node(new_population_island[i])) < self.total_fitness(
                    fitness_loss(population_island[i]), self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])):
                updated_group.append(new_group[i])
                final_population_island[i] = new_population_island[i]
                continue
            else:
                """ 如果旧解的总适应度低新于解 """
                updated_group.append(old_group[i])
                final_population_island[i] = new_population_island[i]
                continue
        return updated_group, final_population_island


    def run_PSO(self):
        #初始化种群和参数[10,4,1,8,1]
        # 初始化种群和参数
        gbest_num=0
        new_pop_X = self.initialization(self.population_x)#初始化个体编号[1,10][1,4][1,1][1,8][1,1]
        print(new_pop_X)
        new_pop_island = np.full((population_size, E_num), 0)  # 用来存放电网状态
        for i in range(population_size):
            new_pop_island[i] = find_island(self.population_change(new_pop_X [i]))
        pbest = self.initialization(self.population_x)
        """pbest代表每个粒子的个体历史最优位置"""
        pbest_island=np.full((population_size,E_num),0)
        for i in range(population_size):
            pbest_island[i]=find_island(self.population_change(pbest[i]))
            """用来存放每个个体历史最优电网状态"""
        record = []
        sum_sum = []

        Vmax = self.get_Vmax(self.bounds)
        Vmin = self.get_Vmin(self.bounds)
        pop_V = self.initialization_V(Vmin,Vmax)
        gbest,gbest_num = self.find_teacher(new_pop_X,new_pop_island)
        for iteration in range(self.iteration_number):
            sum1 = 0
            # 计算所有解的平均值
            """
            改成算划分孤岛后的population_island适应度
            """
            record.append(fitness_loss(new_pop_island[gbest_num]) * 10000)
            for i in range(population_size):
                sum1 = sum1 + fitness_loss(new_pop_island[i]) * 10000
            sum1 = float(sum1 / population_size)
            sum_sum.append(sum1)
            #粒子更新位置
            old_pop_X,old_pop_island= self.update_X(new_pop_X,pop_V,new_pop_island)

            #存储粒子所有迭代最优解
            pbest,pbest_island = self.save_pbest(pbest,pbest_island,old_pop_X,old_pop_island)
            #更新粒子
            new_pop_X=copy.deepcopy(pbest)
            new_pop_island=copy.deepcopy(pbest_island)
            #更新速度
            pop_V = self.update_V(new_pop_X, pop_V, pbest, gbest, Vmin, Vmax)
            #pbest = self.save_pbest(pbest,new_pop_X)
            gbest,gbest_num = self.save_gbest(pbest,pbest_island)#整体最优解
            # print(fitness_loss(new_pop_island[gbest_num]) * 10000)
            # input("程序已输出结果，按回车键继续...")

        final_Solution = pbest_island[gbest_num]
        print(final_Solution)
        print(self.total_fitness(fitness_loss(pbest_island[gbest_num]),self.fitness_switch(pbest_island[gbest_num]),self.fitness_node(pbest_island[gbest_num])))
        island_rate = self.fitness_switch(final_Solution)
        totalss = 0
        for l in sum_sum:
            totalss = totalss + l
        totals = totalss/self.iteration_number
        print(totals)#平均
        return island_rate,gbest,gbest_num,record,sum_sum

start_time = time.time()
result = PSO(x_solutions.create_population(population_size,N,fault_branch),task_number,population_size,iteration_number,bounds,fault_branch)
island_rate,gbest,gbest_num,record,sum_sum= result.run_PSO()
end_time = time.time()
print("PSO_IC")






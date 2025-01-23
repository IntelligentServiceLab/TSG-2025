
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
class TLBO_IC():
    """教学优化算法"""

    def __init__(self, population_x, task_number, population_size, iteration_number,fault_branch):
        self.population_x = population_x  # 初始种群
        self.task_number = task_number  # 环个数
        self.population_size = population_size  # 种群规模
        self.iteration_number = iteration_number  # 迭代次数
        self.fault_brach = fault_branch #故障支路

    #编码转换
    def population_change (self,population_x):
        new_popolation = copy.deepcopy(population_start_change)
        for i in range(task_number):
            j =int(population_x[i]-1)
            x= N[i][j]
            new_popolation[x] = 0
        return new_popolation



    def initialization(self,population_x):
        """初始化阶段:根据种群规模，生成相应个数的个体（服务组合解）;
        通过为每个任务随机挑选候选服务来初始化一个组合服务"""
        population = self.population_x  # 种群
        return population
    def teacher_phase(self, population, teacher,population_island):
        # //该方法的作用是在教师阶段更新种群。每个个体通过与老师和个体平均值的差值来学习，从而向老师靠近。
        """教师阶段:所有个体通过老师和个体平均值的差值像老师;
        学习参数是 种群列表 和 候选服务集的上下界列表"""
        Mean = self.get_Mean(population)  # 种群中每个任务的平均值列表
        old_population = copy.deepcopy(population)  # 保存算法开始前的种群
            # 这个循环遍历每个个体
        for i in range(0, self.population_size):
            # TF = round(1 + random.random())  # 教学因素 = round[1 + rand(0, 1)]
                # r量 = random.random()  # ri=rand(0,1), 学习步长
                # 这个循环与第一个循环一起用来更新每个个体的第j个任务
            for j in range(0, self.task_number):
                TF = round(1 + random.random())  # 教学因素 = round[1 + rand(0, 1)]，TF 作为教学因素，影响个体向老师学习的程度
                r = random.random()  # ri=rand(0,1), 学习步长，r 作为学习步长，控制个体向老师学习时的步长大小。
                 # 更新第i个解的第j个任务的响应时间
                difference_Res = r * (teacher[j] - TF * Mean[j][0])# //计算老师和平均值的差值，然后乘以学习步长和教学因素，得到任务的更新量。
                old_population[i][j] += difference_Res# //将更新加到个体的任务上
                if old_population[i][j] < population_min[j]:
                    old_population[i][j]=population_min[j]
                elif old_population[i][j] > population_max[j]:
                    old_population[i][j]=population_max[j]
                else:
                    x=int(old_population[i][j]/1)
                    y=old_population[i][j]%1
                    if y>=0.5:
                        old_population[i][j]=x+1
                    else:
                        old_population[i][j]=x

                # 在教师阶段方法内直接调用update方法
        new_population,population_island = copy.deepcopy(self.update(population, old_population,teacher,population_island))
        return new_population,population_island

    def min_max_turnone(self,fitness, min_value, max_value):
        normalized_value = (fitness - min_value) / (max_value - min_value)
        return normalized_value

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
    def total_fitness(self, fit_loss, fit_switch, fit_node):
        total_fit = 0
        total_fit = C1 * self.min_max_turnone(fit_loss, min_max_fit_loss[0],
            min_max_fit_loss[1]) - C2 * self.min_max_turnone(fit_switch,min_max_fit_switch[0],min_max_fit_switch[
            1]) - C3 * self.min_max_turnone(fit_node, min_max_fit_node[0], min_max_fit_node[1])
        return total_fit
    #恢复率
    def fitness_node(self, solution):
        loadd = 0
        G = create_adjacency_matrix(branch_node, solution, branch_num)
        component = self.find_connected_components(G)  # 找到连通分量
        components_num = len(component)  # 有几个连通分量
        for i in range(components_num):
            z = 0
            for j in component[i]:
                if j in PS:#连接了电源
                    z = 1
                    continue
            if z == 1:
                for l in component[i]:
                    loadd += load[l]
        num = loadd / total_load
        return num

    def student_phase(self, population,teacher,population_island):
        """学生阶段"""
        old_population = copy.deepcopy(population)  # 保存算法开始前的旧种群
        new_population = []  # 初始化新种群
        for i in range(0, self.population_size):
            #num_list = self.get_list()  # 生成一个包含所有个体索引的列表
            num_list = list(range(0, self.population_size))
            num_list.remove(i) # //从列表中移除当前个体的索引。
            index = random.choice(num_list)  # 这两步获得一个除了自身以外的随机索引.//用于选择一个同学个体

            X = copy.deepcopy(population[i])
            Y = copy.deepcopy(population[index])  # 被选中与X交叉的个体
            # 如果Y支配X, Y比X好
            if self.total_fitness(fitness_loss(population_island[index]),self.fitness_switch(population_island[index]),self.fitness_node(population_island[index]))<self.total_fitness(fitness_loss(population_island[i]),self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])):

                r = random.random()  # 学习步长ri=rand(0,1)
                # //遍历每个任务
                for j in range(0, self.task_number):
                    # 更新第X的第j个任务的响应时间
                    """
                    if j in self.fault_brach:
                        X[j] = 0
                        continue
                    """
                    X[j] += r * (Y[j] - X[j])
                    if X[j] < population_min[j]:
                        X[j] = population_min[j]
                    elif X[j] > population_max[j]:
                        X[j] = population_max[j]
                    else:
                        xx = int(X[j] / 1)
                        yy = X[j] % 1
                        if yy >= 0.5:
                            old_population[i][j] = xx + 1
                        else:
                            old_population[i][j] = xx
                    """
                    if X[j] >= 0.5:
                        X[j] = 1
                    else:
                        X[j] = 0
                    """
            else:
                r = random.random()  # 学习步长ri=rand(0,1)
                # //遍历每个任务
                for j in range(0, self.task_number):
                    # 更新第X的第j个任务的响应时间
                    if j in self.fault_brach:
                        X[j] = 0
                        continue
                    X[j] += r * (X[j] - Y[j])
                    if X[j] < population_min[j]:
                        X[j] = population_min[j]
                    elif X[j] > population_max[j]:
                        X[j] = population_max[j]
                    else:
                        xx = int(X[j] / 1)
                        yy = X[j] % 1
                        if yy >= 0.5:
                            old_population[i][j] = xx + 1
                        else:
                            old_population[i][j] = xx
            new_population.append(X)

        # 在教师阶段方法内直接调用refine方法
        # new_population = copy.deepcopy(self.refine(population, self.bounds))

            # 在教师阶段方法内直接调用update方法
        new_population,population_island = copy.deepcopy(self.update(old_population, new_population,teacher,population_island))

        return new_population,population_island

    def find_teacher(self, population,population_island):
        """
        找到种群中的老师(Pareto解集)
        比较适应度并判断解是否满足约束条件
        """
        min = 0
        for i in range(0, self.population_size):
            if self.total_fitness(fitness_loss(population_island[min]),
                                  self.fitness_switch(population_island[min]),self.fitness_node(population_island[min])) > self.total_fitness(
                    fitness_loss(population_island[i]), self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])):
                min = i
        return population[min],min

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
        num = sum_load/DG_tatol_load
        return num
    def update(self, old_group, new_group,teacher,population_island):
        """这个函数用来更新种群:若新解支配旧解，则替换;否则保留"""
        """
        
        old_population_change = self.population_change(old_group)
        new_group_change = self.population_change(new_group)
        teacher_change = self.population_change(teacher)
        """
        final_population_island = np.full((population_size, E_num), 0)
        new_population_island = np.full((population_size, E_num), 0)
        for i in range(self.population_size):
            new_population_island[i] = find_island(self.population_change(new_group[i]))
        updated_group = [] # //初始化一个空列表，用于存储更新后的种群。
        for i in range(self.population_size):
            # 如果新解支配旧解
            if self.total_fitness(fitness_loss(new_population_island[i]),self.fitness_switch(new_population_island[i]),self.fitness_node(new_population_island[i])) < self.total_fitness(fitness_loss(population_island[i]),self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])):
                """new_group[i][fault_branch[0]]==0and new_group[i][fault_branch[1]]==0and new_group[i][fault_branch[2]]==0 and """
                updated_group.append(new_group[i])
                final_population_island[i] = new_population_island[i]
                continue
            """
            //如果新解的总适应度高于旧解，或者新旧解适应度相同但旧解是老师，则保留旧解。
            """
            if self.total_fitness(fitness_loss(new_population_island[i]),self.fitness_switch(new_population_island[i]),self.fitness_node(new_population_island[i])) > self.total_fitness(fitness_loss(population_island[i]),self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])):
                updated_group.append(old_group[i])
                final_population_island[i] = population_island[i]
                continue
            if self.total_fitness(fitness_loss(new_population_island[i]),self.fitness_switch(new_population_island[i]),self.fitness_node(new_population_island[i])) == self.total_fitness(fitness_loss(population_island[i]),self.fitness_switch(population_island[i]),self.fitness_node(population_island[i])):
                if np.array_equal(teacher,old_group[i]):
                    updated_group.append(old_group[i])
                    final_population_island[i] = population_island[i]
                    continue
                updated_group.append(old_group[i])
                final_population_island[i] = population_island[i]
                continue

        return updated_group,final_population_island
    def get_list(self):
        """"为了学生阶段获得一个种群大小的数字列表"""
        nums_list = []
        for i in range(0, self.population_size):
            nums_list.append(i)
        return nums_list
    def get_Mean(self, population):
        """获得种群中 每个任务 的平均值;
           参数为种群;
           返回值为每个任务平均值的列表
        """
        Mean = []
        #population的每一列i，与每一行
        for i in range(0, self.task_number):

            Sum_Res = np.zeros(37)
            Mean_i = []
            for j in range(0, self.population_size):

                Sum_Res[i] += population[j][i]

            Mean_i.append(Sum_Res[i] / self.population_size)

            Mean.append(Mean_i)

        return Mean
    def run_TLBO_IC(self):
        #初始化种群和参数

        # //初始化种群和参数
        new_population = self.initialization(self.population_x)
        """
        存放种群最优的孤岛划分的解
        """
        population_island = np.full((population_size, E_num), 0)
        for i in range(population_size):
            population_island[i] = find_island(self.population_change(new_population[i]))
        # //从初始化的种群中找到最优解，即老师
        teacher_solution,teacher_num = self.find_teacher(new_population,population_island)
        # teacher_solution_ic = self.population_ic(teacher_solution, task_number)
        # //创建两个空列表record和sum_sum，用于记录每一代的最优适应度值和所有解的平均适应度值

        record=[]
        sum_sum = []

        for iteration in range(self.iteration_number):
            sum1 = 0

            record.append(fitness_loss(population_island[teacher_num]) * 10000)
            for i in range(population_size):
                sum1 = sum1+fitness_loss(population_island[i])*10000
            sum1=float(sum1/population_size)
            sum_sum.append(sum1)
            # 教师阶段
            old_population,population_island=self.teacher_phase(new_population, teacher_solution,population_island)
            # 学生阶段66
            new_population,population_island=self.student_phase(old_population,teacher_solution,population_island)
            # 更新最优解
            teacher_solution,teacher_num = self.find_teacher(new_population,population_island)
        #final_Solution = population_island[teacher_num]
        # 输出教师作为最终结果
        island_rate = self.fitness_switch(population_island[teacher_num])
        return island_rate,teacher_solution,population_island[teacher_num],record,new_population,sum_sum
start_time = time.time()
result=TLBO_IC(x_solutions.create_population(population_size,N,fault_branch),task_number,population_size,iteration_number,fault_branch)

island_rate,teacher,teacher_island,record,end_pop,sum_sum=result.run_TLBO_IC()
end_time = time.time()
print("TLBO_IC")





def read_txt_to_matrix(file_path,type,delimiter=' '):
    """
    读取txt文件中的元素并生成矩阵。

    参数:
    file_path -- txt文件的路径
    delimiter -- 数据之间的分隔符，默认为空格

    返回:
    一个NumPy矩阵，包含文件中的数据
    """
    matrix_data = []  # 初始化一个空列表来存储所有行的数据

    # 打开文件并读取所有行
    with open(file_path, 'r') as file:
        for line in file:
            # 移除行尾的换行符并分割行成元素列表
            row_data = line.strip().split(delimiter)
            # 将元素转换为浮点数并添加到列表中
            matrix_data.append([type(num) for num in row_data])

    # 使用 NumPy 将列表转换为矩阵
    return np.array(matrix_data)

RX=read_txt_to_matrix('支路连接情况.txt',int,delimiter=' ')

switch_line=[[20,7],
    [8,14],
    [11,21],
    [17,32],
    [24,28]]

edges = np.concatenate((RX, switch_line), axis=0) # edges 通过合并 RX 和 switch_line 创建了一个包含所有节点连接的数组。
NODE_COPY_n=edges.astype(int) # NODE_COPY_n 是 edges 的整数类型拷贝。

#branch_node 和 reversed_branch_node 分别定义了电网的正向和反向分支节点。
branch_node=[[ 0 ,1],[ 1 ,2],[ 2 , 3], [ 3, 4], [ 4, 5], [ 5, 6], [ 6, 7], [ 7, 8], [ 8, 9], [ 9,10], [10,11], [11,12],
             [12,13], [13,14], [14,15], [15,16], [16,17], [ 1,18], [18,19], [19,20], [20,21], [ 2,22], [22,23], [23,24],
             [ 5,25],[25,26], [26,27], [27,28], [28,29], [29,30], [30,31],[31,32],[20, 7],[ 8,14],[11,21],[17,32],[24,28]]
reversed_branch_node = [[ 1 , 0], [ 2 , 1], [ 3 , 2], [ 4 , 3], [ 5 , 4], [ 6 , 5], [ 7 , 6], [ 8 , 7], [ 9 , 8], [10 , 9],
                        [11 ,10], [12 ,11], [13 ,12], [14 ,13], [15 ,14], [16 ,15], [17 ,16], [18 , 1], [19 ,18], [20 ,19],
                        [21 ,20], [22 , 2], [23 ,22], [24 ,23], [25 , 5], [26 ,25], [27 ,26], [28 ,27], [29 ,28], [30 ,29],
                        [31 ,30], [32 ,31], [ 7 ,20], [14  ,8], [21 ,11], [32, 17], [28 ,24]]
graphs=[[0] * 33 for _ in range(33)] # 邻接矩阵
# 初始化邻接矩阵
for node_i in range(37):
    node1=int(edges[node_i][0])
    node2=int(edges[node_i][1])
    graphs[node1][node2] = 1
    graphs[node2][node1] = 1  #一个33节点矩阵图

start_graphs=copy.deepcopy(graphs)
x=[]



def create_population(population_size,N,fault_branch):
    population = np.ones((population_size,len(N)))

    for i in range(population_size):
        for j, row in enumerate(N):
            # 获取当前行非零元素的列下标
            non_zero_elements = [index for index, value in enumerate(row) if value != 0]
            # 如果当前行有不为0的元素，则随机选择一个
            if non_zero_elements:
                selected_elements = random.choice(non_zero_elements)
                population[i][j] = selected_elements + 1

    return population





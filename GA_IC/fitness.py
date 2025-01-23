


def dfs(graph, node, visited, component):
    # 标记当前节点为已访问
    visited.add(node)
    component.append(node)

    # 遍历所有相邻节点
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, component)

"""
找连通分量
"""
def find_connected_components(graph):
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

"""
得到指定连通分量的Bus和Branch
"""
def find_Bus_Branch(node,line,components):
    nde = copy.deepcopy(node)
    lne = copy.deepcopy(line)

    component = sorted(components)
    filtered_node = [row for row in nde if row[0] in component] #去除其他的节点
    """
    重新编号
    """
    for i in range(len(filtered_node)):
        filtered_node[i][0] = i
    filtered_lines = [line for line in lne if line[1] in component and line[2] in component] #去除其他线路
    """
    重新编号
    """
    for i in range(len(filtered_lines)):
        index_1 = component.index(filtered_lines[i][1])
        index_2 = component.index(filtered_lines[i][2])
        filtered_lines[i][0] = i + 1
        filtered_lines[i][1] = index_1
        filtered_lines[i][2] = index_2
    return filtered_node,filtered_lines

"""
去除未用线路
"""
def remove_lines(xx, linessss):
    linesss = copy.deepcopy(linessss)
    for i in range(len(xx)):
        if xx[i] == 0:
            linesss[i][0] = 0
    filtered_matrix = [row for row in linesss if row[0] != 0]

    return filtered_matrix


def fitness_loss(x):
    V_Max = 13.86
    V_Min = 11.34
    lines = remove_lines(x,liness)
    branch_num = 33

    G = create_adjacency_matrix(node_branch, x, branch_num)#图
    component = find_connected_components(G)#找到连通分量
    components_num = len(component) #有几个连通分量
    fit_loss = [ ] #用来存储每个分量的损耗
    for m in range(components_num):

        node = copy.deepcopy(nodes)
        line = copy.deepcopy(lines)

        Bus,Branch = find_Bus_Branch(node,line,component[m])
        if Branch == []:
            break

        busnum, row = np.shape(Bus)  # busnum 33节点数,row 列数
        branchnum, row = np.shape(Branch)  # branchnum 32
        soubus = np.array(Branch)[:, 1]  # 支路初始节点序号
        mobus = np.array(Branch)[:, 2]  # 支路终节点序号

        Vbus = np.ones((busnum, 1))  # 初始化节点电压
        Vbus[:, 0] = 12.66  # 电压初始化为12.66
        Vbus1 = Vbus.copy()  # 下一代的节点电压
        Ploss = np.zeros((busnum, busnum))  # busnum33 #存储P损耗
        Qloss = np.zeros((busnum, busnum))  # 存储Q损耗
        T1 = []  # shape(32,5 )#存放支路
        T2 = []  # shape(32,5 )#存放支路

        e = 1
        k = 0
        Branch1 = Branch.copy()  # 【支路序号、初始节点、终节点、R,X]
        n = 0
        while np.array(Branch1).shape[0] != 0:
            m = 0
            s, row = np.shape(Branch1)  # s=32,row=5  s为支路数

            while s > 0:
                # 找到起始节点和Branch1[s, 2] 相等的序号  s-1为最后一个支路序号
                # （python是从0开始计数，所以s-1为最后一个支路序号
                t = np.where(np.array(Branch1)[:, 1] == np.array(Branch1)[s - 1, 2])[0]

                if t.size == 0:  # 即不存在一个节点（即是初始节点 又是终节点）的节点，这个节点仅仅只是终节点，即叶子节点
                    T1.append(Branch1[s - 1][:])  # 把叶子节点存放到T1中
                else:
                    T2.append(Branch1[s - 1][:])  # 该s-1节点 非叶子节点 。存放到T2中
                s = s - 1  # 修改支路序号

            Branch1 = T2
            T2 = []

        PP = np.zeros((busnum, busnum))  # 存放支路P  shape(33,33)
        QQ = np.zeros((busnum, busnum))  # 存放支路Q  shape(33,33)
        for i in tqdm(range(3)):
            P = np.zeros((busnum, 1))  # shape(33,1) #存放节点P
            Q = np.zeros((busnum, 1))  # shape(33,1)  #存放节点Q
            for s in range(branchnum):  # 遍历每一条支路

                i = T1[s][1]  # 支路首端节点（靠近根节点那一侧）
                j = T1[s][2]  # 支路终节点（靠近负荷侧节点）
                R = T1[s][3]  # 支路R
                X = T1[s][4]  # 支路Q
                Pload = Bus[j][1]  # 支路终节点 负荷P （靠近负荷侧节点）
                Qload = Bus[j][2]  # 支路终节点 负荷Q （靠近负荷侧节点）
                ###计算沿线电流  (P^2+Q^2)/U^2 ， PQU 为支路终节点的（靠近负荷侧）
                II = (np.power((Pload + P[j]), 2) + np.power((Qload + Q[j]), 2)) / np.power((Vbus[j] * 1000), 2)

                # 计算线路上的损耗
                Ploss[i, j] = II * R;
                Qloss[i, j] = II * X;
                # 支路P= 支路网损P+与它相连接的节点P（靠近负荷侧）【不断修改支路P，就是不断添加相连节点j的P
                # 这里Pload+P[j]=与它相连接的节点j的P（靠近负荷侧）
                # 支路Q 同理

                PP[i, j] = Pload + Ploss[i, j] + P[j]
                QQ[i, j] = Qload + Qloss[i, j] + Q[j]
                # 修改支路首端节点P（靠近根节点） 。支路首端节点P=支路首端节点P+与该节点相连支路P （相连支路可能不只1条）
                P[i] = P[i] + PP[i, j]
                Q[i] = Q[i] + QQ[i, j]
                ####前推

            for s in range(branchnum - 1, -1, -1):  # 遍历支路
                i = T1[s][1]  # 支路首端节点（靠近根节点那一侧）
                j = T1[s][2]  # 支路终节点（靠近负荷侧节点）
                R = T1[s][3]  # 支路R
                X = T1[s][4]  # 支路Q

                # 支路ij 电压降落纵分量 dv1=(PR+QX)/U  ,这里的PQ 为支路ij 的PQ，不能是节点的，因为一个节点可能含有多个支路

                dv1 = ((PP[i, j] * R + QQ[i, j] * X) / (Vbus[i] * 1000))[0]
                ##支路ij 电压降落横分量 dv2=(PX-QR)/U  ,这里的PQ 为支路ij 的PQ，不能是节点的，因为一个节点可能含有多个支路
                dv2 = ((PP[i, j] * X - QQ[i, j] * R) / (Vbus[i] * 1000))[0]
                ##
                Vbus[j] = np.sqrt(np.power((Vbus[j] - dv1), 2) + np.power(dv2, 2))  # 修改前推终节点(负荷侧）电压

            e = max(abs(Vbus1 - Vbus))
            Vbus1 = Vbus

        if  all(value > V_Min for value in Vbus1) and all(value < V_Max for value in Vbus1):
            fit_loss_tatal = 0
        else:
            fit_loss_tatal = 10
        fit_loss_tatal =  fit_loss_tatal+sum(num for row in Ploss for num in row if num != 0)
        fit_loss.append(fit_loss_tatal)

    sumF_loss = 0
    for i in range(len(fit_loss)):
        sumF_loss = sumF_loss + fit_loss[i]
    return sumF_loss


"""
从txt中读取数据生成矩阵
"""
def read_txt_to_matrix(file_path, delimiter=' '):
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
            matrix_data.append([float(num) for num in row_data])

    # 使用 NumPy 将列表转换为矩阵
    return np.array(matrix_data)







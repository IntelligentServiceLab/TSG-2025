

B_num = 33
Pc=[]# 分布式能源充电/放电功率
num = 20 #划分孤岛的次数
branch_num = 33 #节点数
#island = [num][B_num+2] # 存储孤岛的地方，最后一位存储适应度，倒数第二位判断是否计算过适应度
all_nodes = list(range(33)) #所有的节点编号
"""
没有断路
branch_node = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],  [10, 11], [11, 12],
                   [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21], [2, 22],
                   [22, 23], [23, 24],[5, 25], [25, 26], [26, 27], [27, 28], [29, 30], [30, 31], [31, 32],
                   [20, 7], [8, 14],[11, 21], [17, 32], [24, 28],[24,28]]
"""
branch_node = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
                   [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21], [2, 22],
                   [22, 23], [23, 24],
                   [5, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [20, 7], [8, 14],
                   [11, 21], [17, 32], [24, 28]]
"""
广度优先遍历
"""
def bfs_search(xx,visited,sum,island_node,DG_load):

    search_queue = deque()
    x = xx.copy()
    for i in range(len(island_node)):
        search_queue.append(island_node[i])
    #Y = [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1]
    Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    G_all = create_adjacency_matrix(node_branch, Y, branch_num)
    island_G = create_adjacency_matrix(node_branch, x, branch_num)
    while search_queue:
        # 弹出队首节点
        current_node = search_queue.popleft()

        # 遍历所有相邻节点
        for i in range(len(island_G[current_node])):


            if island_G[current_node][i] != 0 and not visited[i]:
                # 如果相邻节点未被访问过且有连边，则将其加入搜索队列
                #remaining_nodes.remove(i)
                G_2 = island_G.copy()
                for m in range(33):
                    island_G[i][m] = 0
                    island_G[m][i] = 0
                connected_nodes = list(nx.node_connected_component(nx.from_numpy_array(island_G), 0)) #主网
                remaining_nodes = list([node for node in all_nodes if node not in connected_nodes]) #其他节点
                smm = 0
                for node in remaining_nodes:
                    smm = smm + load[node]
                #if set(remaining_nodes) == set(connected_nodes) and DG_load > sum + load[i]:#把这个节点划分给孤岛后主网依然连通,而且满足约束
                if DG_load > smm:
                    sum = smm
                    result = []
                    result = list(set(remaining_nodes) - set(island_node))
                    #search_queue.append(i)
                    for element in result:
                        search_queue.append(element)
                        visited[element] = True
                    island_node = remaining_nodes

                island_G =G_2

    return island_node

"""
判断孤岛外的节点是否都连接，如果有散落节点看是否全部能划分到孤岛中，如果不能就放弃以这个分布式电源划分孤岛，start_node是孤岛节点,l是DG_node的下标
"""
def is_all_nodes_connected(G_old, xx, l, DG_node):
    """
    断开孤岛旁边的节点
    """
    x = xx.copy()
    G = G_old.copy()
    visited = [0]*branch_num
    for i in range(branch_num):
        G[DG_node][i] = 0
        G[i][DG_node] = 0
    connected_nodes = list(nx.node_connected_component(nx.from_numpy_array(G), 0))
    remaining_nodes = list([node for node in all_nodes if node not in connected_nodes])
    sum = 0
    for j in remaining_nodes:
        sum = sum + load[j]
        visited[j] = 1
    if DG_load[l] < sum:
        return [0]
    else:
        add_island = bfs_search(x,visited,sum,remaining_nodes,DG_load[l])
        return add_island
"""
得到划分孤岛后的X,island_list是孤岛
"""
def get_X(G,island_list,is_island,xx):
    x = xx.copy()
    for i in range(len(island_list)):
        if is_island[i] == 0: #孤岛未启用
            continue
        remaining_nodes = [node for node in all_nodes if node not in island_list[i]] #除孤岛外的所有节点
        for idx, edge in enumerate(branch_node): #
            # 检查这条边的两个节点是否分别来自孤岛和其他节点，branch_node是配电网中所有的边
            if (edge[0] in island_list[i] and edge[1] in remaining_nodes) or (edge[0] in remaining_nodes and edge[1] in island_list[i]):
                x[idx] = 0
    return x
"""
找到最优的孤岛
"""
def find_island(x):
    """
    生成邻接矩阵
    """
    # branch_node 和 reversed_branch_node 分别定义了电网的正向和反向分支节点。
    branch_node_1 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
                   [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21], [2, 22],
                   [22, 23], [23, 24],
                   [5, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [20, 7], [8, 14],
                   [11, 21], [17, 32], [24, 28]]
    for i in range(len(x)):
        if x[i] == 0:
            branch_node_1[i] = [0,0]
    G = create_adjacency_matrix(node_branch, x, branch_num)
    """
    划分所有孤岛
    """
    island_list = [[0] for _ in range(len(DG_nodes))]
    is_island = [1]*len(DG_nodes) #表示该孤岛划分是否使用
    for i in range(len(DG_nodes)):
        if is_all_nodes_connected(G, x, i, DG_nodes[i]) == [0]: #该孤岛不启用
            is_island[i] = 0
            continue
        island_list[i] = is_all_nodes_connected(G, x, i, DG_nodes[i])
    """
    看是否有孤岛重合，有则孤岛融合
    """
    for i in range(len(DG_nodes)):
        for j in range(i + 1, len(DG_nodes)):  # 遍历每一对不同的子列表
            if is_island[i] ==0 or is_island[j] ==0: #有一个孤岛不启用
                continue
            common_elements = set(island_list[i]) & set(island_list[j])  # 计算交集

            is_adjacent = 0
            for node1 in island_list[i]:
               for node2 in island_list[j]:
                  if [node1,node2] in branch_node_1 or [node2,node1] in branch_node_1:
                    G[node1][node2] = 1
                    G[node2][node1] = 1
                    is_adjacent = 1
                    break
               else:
                    is_adjacent = 0

               if is_adjacent == 1:
                    break
            if common_elements or is_adjacent: #有交集或者相邻,则融合，把前面的孤岛融合到后面那个去,后面的孤岛重新广度优先
                island_list_1 = list(set(island_list[i]) | set(island_list[j])) #更新后面那个孤岛的元素
                visited_1 = [0] * branch_num
                summ = 0
                for z in island_list_1:
                    if z == 0:
                        continue
                    visited_1[z] = 1
                    summ = summ + load[z]
                island_list[j] = bfs_search(x,visited_1, summ, island_list_1, DG_load[i] + DG_load[j])
                island_list[i] = [0]
                is_island[i] = 0 #第一个孤岛放弃
                break
    X = get_X(G, island_list, is_island, x) #孤岛划分后的X

    return X






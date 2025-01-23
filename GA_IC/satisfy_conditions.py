
"""
生成邻接矩阵
"""
def create_adjacency_matrix(connection_matrix,x,branch_num):

    # 创建n x n的全零邻接矩阵
    adjacency_matrix = np.zeros((branch_num, branch_num))
    num = -1
    # 填充邻接矩阵
    for i, j in connection_matrix:
        num = num + 1
        if x[num] == 0:
            continue
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    return adjacency_matrix

#读取txt文件中的数据
def read_txt_to_matrix(file_path, delimiter=' '):
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


#功率约束
def power_constraint(Peqi, PLit, PlossI):
    sum_PLit = sum(PLit)  # 计算L中所有元素的和
    constraint = Peqi - sum_PLit - PlossI  # 计算功率约束
    return constraint
def check_current_constraint(I_ijx, I_ij_max):
    # 检查电流约束是否满足
    if I_ijx <= I_ij_max:
        return True
    else:
        return False

def check_voltage_constraint(U_i_t, U_i_min, U_i_max):
    # 检查电压约束是否满足
    if U_i_min <= U_i_t <= U_i_max:
        return True
    else:
        return False



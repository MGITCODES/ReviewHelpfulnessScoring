import numpy as np


def user_quality_scoring(data_set):
	f = open(data_set, 'r')
	edges = [line.strip('\n').split(' ') for line in f]
	# print(edges)

	# the set of nodes
	nodes = []
	for edge in edges:
		if edge[0] not in nodes:
			nodes.append(edge[0])
		if edge[1] not in nodes:
			nodes.append(edge[1])
	# print(nodes)

	N = len(nodes)

	# nodes 2 index
	i = 0
	node_to_num = {}
	for node in nodes:
		node_to_num[node] = i
		i += 1
	for edge in edges:
		edge[0] = node_to_num[edge[0]]
		edge[1] = node_to_num[edge[1]]
	# print(edges)

	# Matrix S
	S = np.zeros([N, N])
	for edge in edges:
		S[edge[1], edge[0]] = 1
	# print(S)

	# normalization
	for j in range(N):
		sum_of_col = sum(S[:, j])
		for i in range(N):
			S[i, j] /= sum_of_col
	# print(S)

	# Matrix A
	alpha = 0.85
	A = alpha * S + (1 - alpha) / N * np.ones([N, N])
	# print(A)

	# user quality score -> Qu_n
	Qu_n = np.ones(N) / N
	Qu_n1 = np.zeros(N)

	e = 100000  # 误差初始化
	k = 0  # 记录迭代次数
	# print('loop...')

	while e > 0.00000001:  # 开始迭代
		Qu_n1 = np.dot(A, Qu_n)  # 迭代公式
		e = Qu_n1 - Qu_n
		e = max(map(abs, e))  # 计算误差
		Qu_n = Qu_n1
		k += 1
		# print('iteration %s:' % str(k), Qu_n1)

	return Qu_n1


# test
# if __name__ == '__main__':
# 	dataset = 'input.txt'
# 	# userqualityscoring(dataset)
# 	print('final result:', user_quality_scoring(dataset))



import numpy as np
from user_quality_scoring_model import user_quality_scoring

def review_helpfulness_scoring(data_set):
	f = open(data_set, 'r')
	edges = [line.strip('\n').split(' ') for line in f]
	print(edges)

	# the set of nodes
	nodes = []
	for edge in edges:
		if edge[0] not in nodes:
			nodes.append(edge[0])
		if edge[1] not in nodes:
			nodes.append(edge[1])
	print(nodes)

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
	print(edges)

	# Matrix S
	S = np.zeros([N, N])
	for edge in edges:
		S[edge[1], edge[0]] = 1
	print(S)

	# normalization
	for j in range(N):
		sum_of_col = sum(S[:, j])
		for i in range(N):
			S[i, j] /= sum_of_col
	print(S)

	# Matrix A
	beta = 0.85
	#从这里开始改
	user_quality_list = user_quality_scoring(data_set)
	array_list = np.array(user_quality_list)
	I_quw = np.diag(array_list)
	A = I_quw*(beta * S + (1 - beta) / N * np.ones([N, N]))
	print(A)

	# review helpfulness score -> Hr_n
	Hr_n = np.ones(N) / N
	Hr_n1 = np.zeros(N)

	e = 100000  # 误差初始化
	k = 0  # 记录迭代次数
	print('loop...')

	while e > 0.00001:  # 开始迭代
		Hr_n1 = np.dot(A, Hr_n)  # 迭代公式
		e = Hr_n1 - Hr_n
		e = max(map(abs, e))  # 计算误差
		Hr_n = Hr_n1
		k += 1
		print('iteration %s:' % str(k), Hr_n1)

	return Hr_n1

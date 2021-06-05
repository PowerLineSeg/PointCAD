import numpy as np

def main():
	Instance_IoU = np.loadtxt("Instance_IoU.txt")
	#print(Instance_IoU.shape)
	#Instance_IoU = Instance_IoU[:10]
	
	Ins_Num = Instance_IoU.shape[0]
	
	sorted_idx = np.argsort(np.argsort(Instance_IoU))
	sorted_idx = Ins_Num - sorted_idx
	
	Total_Prob = np.sum(sorted_idx)
	split_Prob = np.cumsum(sorted_idx)
	
	split_Prob = np.expand_dims(split_Prob, axis=0)
	split_Prob = np.tile(split_Prob,(Ins_Num,1))
	
	selected_sample = np.random.uniform(0,Total_Prob,Ins_Num)
	selected_sample = np.expand_dims(selected_sample, axis=1)
	selected_sample = np.tile(selected_sample,(1,Ins_Num))
	
	selected_sample = ((selected_sample - split_Prob) > 0).astype(int)
	
	#print(selected_sample)
	indices = []
	for idx_i in range(Ins_Num):
		for idx_j in range(Ins_Num):
			if selected_sample[idx_i,idx_j] == 0:
				indices.append(idx_j)
				break
	
	return iter(indices)
	#print(iter(indices))
		

if __name__ == "__main__":
	main()
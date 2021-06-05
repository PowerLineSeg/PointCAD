import torch
import numpy as np
from torch.utils.data import Sampler

class BalancedRandomSampler(Sampler):
    r"""This sampler is responsible for creating balanced batch based on the class distribution.
    It is implementing a replacement=True strategy for indices selection
    """
    def __init__(self, labels, replacement=True):

        self.num_samples = len(labels)

        self.idx_classes, self.counts = np.unique(labels, return_counts=True)
        self.indices = {
           idx: np.argwhere(labels == idx).flatten() for idx in self.idx_classes
        }

    def __iter__(self):
        indices = []
        for _ in range(self.num_samples):
            idx = np.random.choice(self.idx_classes)
            indice = int(np.random.choice(self.indices[idx]))
            indices.append(indice)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        return "{}(num_samples={})".format(self.__class__.__name__, self.num_samples)

class Hard_Mining_Sampler(Sampler):
	"""
	This Sampler is used in the Hard Mining of SemanticSegmentation
	"""
	
	def __init__(self, hard_mining_status):
		self.iteration = hard_mining_status.Sample_Num
		self.hard_mining_status = hard_mining_status
	
	def __iter__(self):
		Instance_IoU = self.hard_mining_status.Instance_IoU
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
		
		indices = []
		for idx_i in range(Ins_Num):
			for idx_j in range(Ins_Num):
				if selected_sample[idx_i,idx_j] == 0:
					indices.append(idx_j)
					break
		
		return iter(indices)
	
	def __len__(self):
		return self.iteration
		
	def __repr__(self):
		return "Hard_Mining_Sampler, With %d Classes and %d Samples"%(self.hard_mining_status.Class_Num, self.hard_mining_status.Sample_Num)


class Hard_Mining_Contrastive_Learning_Sampler(Sampler):
	"""
	This Sampler is used in the Hard Mining of SemanticSegmentation
	"""
	
	def __init__(self, hard_mining_status):
		self.iteration = hard_mining_status.Sample_Num // 3
		self.hard_mining_status = hard_mining_status
	
	def __iter__(self):
		indices = []
		for idx in range(self.iteration):
			break
		
		
		return None
	
	def __len__(self):
		return self.iteration
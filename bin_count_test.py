import numpy as np

def main():
	ground_truth_vec = [0,1,2,3,4,5,6,7,8,9]
	predicted = [1,2,2,3,4,5,6,7,8,9]
	ground_truth_vec = np.array(ground_truth_vec)
	predicted = np.array(predicted)
	
	number_of_labels = 10
	
	batch_confusion = np.bincount(number_of_labels * ground_truth_vec.astype(int) + predicted, minlength=number_of_labels ** 2).reshape(number_of_labels, number_of_labels)
	TP_plus_FN = np.sum(batch_confusion, axis=1)
	
	print(TP_plus_FN)
	
	print(batch_confusion)
	

if __name__ == "__main__":
	main()
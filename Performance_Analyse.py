import numpy as np
import os
import matplotlib.pyplot as plt

def main():
	Log_path  = "Log_performance"
	filenames = sorted(os.listdir(Log_path))
	Class_IoU_filenames = []
	Instance_IoU_filenames = []
	
	for file in filenames:
		if "Class" in file:
			Class_IoU_filenames.append(file)
		elif "Instance" in file:
			Instance_IoU_filenames.append(file)
		else:
			print("Error in filename: %s"%(file))
			exit()
	
	'''
	print(Class_IoU_filenames)
	print("\n\n\n\n\n")
	print(Instance_IoU_filenames)
	'''
	
	Class_IoU = []
	for epoch in Class_IoU_filenames:
		epoch_filename = os.path.join(Log_path,epoch)
		cur_epoch_class_iou = np.loadtxt(epoch_filename)
		Class_IoU.append(cur_epoch_class_iou)
	Class_IoU = np.array(Class_IoU)
	Epoch_Class_Var = np.var(Class_IoU,axis=1)
	Epoch_Class_Mean = np.mean(Class_IoU,axis=1)	
	
	Instance_IoU = []
	for epoch in Instance_IoU_filenames:
		epoch_filename = os.path.join(Log_path,epoch)
		cur_epoch_instance_iou = np.loadtxt(epoch_filename)
		Instance_IoU.append(cur_epoch_instance_iou)
	Instance_IoU = np.array(Instance_IoU)
	Epoch_Instance_Var = np.var(Instance_IoU,axis=1)
	Epoch_Instance_Mean = np.mean(Instance_IoU,axis=1)	
	
	
	#Plot the Graph
	plt.figure("Instance_IoU_Variance")
	plt.plot(np.arange(Epoch_Instance_Var.shape[0]),Epoch_Instance_Var,'g',label='Instance_IoU_Variance')
	plt.xlabel('Epoch')
	plt.ylabel('Variance')
	plt.legend()
	
	plt.figure("Class_IoU_Variance")
	plt.plot(np.arange(Epoch_Class_Var.shape[0]),Epoch_Class_Var,'r',label='Class_IoU_Variance')
	plt.xlabel('Epoch')
	plt.ylabel('Variance')
	plt.legend()
	
	plt.figure("Mean_IoU")
	plt.plot(np.arange(Epoch_Instance_Mean.shape[0]),Epoch_Instance_Mean,'r',label='Instance_IoU_Mean')
	plt.plot(np.arange(Epoch_Class_Mean.shape[0]),Epoch_Class_Mean,'g',label='Class_IoU_Mean')
	plt.xlabel('Epoch')
	plt.ylabel('MIoU')	
	plt.legend()
	
	
	plt.show()
	
if __name__ == "__main__":
	main()
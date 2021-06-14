import numpy as np
import os
import matplotlib.pyplot as plt

def get_config_result(Log_path):
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
	
	return Epoch_Class_Var,Epoch_Class_Mean,Epoch_Instance_Var,Epoch_Instance_Mean
	
def plot_result(Epoch_Class_Var, Epoch_Class_Mean, Epoch_Instance_Var, Epoch_Instance_Mean):
	#Plot the Graph
	plt.figure("Instance_IoU_Variance")
	for idx in range(len(Epoch_Instance_Var)):
		color = np.random.rand(1,3).tolist()[0]
		plt.plot(np.arange(Epoch_Instance_Var[idx].shape[0]),Epoch_Instance_Var[idx],c=color,label="Instance_IoU_Variance_Config_%d"%(idx))
	plt.xlabel('Epoch')
	plt.ylabel('Variance')
	plt.legend()
	
	plt.figure("Class_IoU_Variance")
	for idx in range(len(Epoch_Class_Var)):
		color = np.random.rand(1,3).tolist()[0]
		plt.plot(np.arange(Epoch_Class_Var[idx].shape[0]),Epoch_Class_Var[idx],c=color,label="Class_IoU_Variance_Config_%d"%(idx))
	plt.xlabel('Epoch')
	plt.ylabel('Variance')
	plt.legend()
	
	plt.figure("Mean_IoU")
	for idx in range(len(Epoch_Instance_Mean)):
		color = np.random.rand(1,3).tolist()[0]
		plt.plot(np.arange(Epoch_Instance_Mean[idx].shape[0]),Epoch_Instance_Mean[idx],c=color,label="Epoch_Instance_Mean_Config_%d"%(idx))

	for idx in range(len(Epoch_Class_Mean)):
		color = np.random.rand(1,3).tolist()[0]
		plt.plot(np.arange(Epoch_Class_Mean[idx].shape[0]),Epoch_Class_Mean[idx],c=color,label="Epoch_Class_Mean_Config_%d"%(idx))
	plt.xlabel('Epoch')
	plt.ylabel('MIoU')	
	plt.legend()
	
	plt.show()
	
def main():	
	Log_path  = ["Log_performance", "Log_performance_instance_hard_mining"]
	
	Epoch_Class_Var = []
	Epoch_Class_Mean = []
	Epoch_Instance_Var = []
	Epoch_Instance_Mean = []
	
	for idx, path in enumerate(Log_path):
		Cur_Class_Var,Cur_Class_Mean,Cur_Instance_Var,Cur_Instance_Mean = get_config_result(path)
		Epoch_Class_Var.append(Cur_Class_Var)
		Epoch_Class_Mean.append(Cur_Class_Mean)
		Epoch_Instance_Var.append(Cur_Instance_Var)
		Epoch_Instance_Mean.append(Cur_Instance_Mean)
	
	plot_result(Epoch_Class_Var, Epoch_Class_Mean, Epoch_Instance_Var, Epoch_Instance_Mean)
	
if __name__ == "__main__":
	main()
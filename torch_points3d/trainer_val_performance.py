import os
import sys
import copy
import torch
import hydra
import time
import logging
import numpy as np

from tqdm.auto import tqdm
import wandb

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS
from torch_points3d.utils.wandb_utils import Wandb
from torch_points3d.visualization import Visualizer

log = logging.getLogger(__name__)


class Hard_Mining_Status:
	"""
	This Class is used to save the training status of current epoch
	"""
	
	def __init__(self, Class_Num=19, Sample_Num=1800):
		self.Class_Num = Class_Num
		self.Sample_Num = Sample_Num
		
		self.Conf_Mat = np.zeros([self.Class_Num,self.Class_Num])
		self.Class_IoU = np.zeros([self.Class_Num,1])
		self.Instance_IoU = np.zeros([self.Sample_Num,1])
		self.Instance_Class_IoU = np.zeros([self.Sample_Num,self.Class_Num])
		
		self.root_path = os.path.abspath(os.path.join(sys.argv[0],"../Log_performance"))
		
		if not os.path.exists(self.root_path):
			os.makedirs(self.root_path)
	
	def __str__(self):
		print(self.Conf_Mat)
		print(self.Class_IoU)
		print(self.Instance_IoU)
		print("mIoU = %.3f"%(np.mean(self.Instance_Class_IoU)))
		return ' '
	
	def reset(self):
		self.Conf_Mat -= self.Conf_Mat
		self.Class_IoU -= self.Class_IoU
		self.Instance_IoU -= self.Instance_IoU
		self.Instance_Class_IoU -= self.Instance_Class_IoU
		
	def set(self,Conf_Mat,Class_IoU,Instance_IoU,Instance_Class_IoU,Epoch=-1):
		self.reset()
		self.Conf_Mat = Conf_Mat
		self.Class_IoU = Class_IoU
		self.Instance_IoU = Instance_IoU
		self.Instance_Class_IoU = Instance_Class_IoU
		
		Ins_IoU_filename = os.path.join(self.root_path, "Instance_IoU_Epoch_%04d.txt"%(Epoch))
		Class_IoU_filename = os.path.join(self.root_path, "Class_IoU_Epoch_%04d.txt"%(Epoch))
		
		if Epoch>=0:
			np.savetxt(Ins_IoU_filename,self.Instance_IoU,fmt="%.4f")
			np.savetxt(Class_IoU_filename,self.Class_IoU,fmt="%.4f")		
		

class Trainer:
    """
    TorchPoints3d Trainer handles the logic between
        - BaseModel,
        - Dataset and its Tracker
        - A custom ModelCheckpoint
        - A custom Visualizer
    It supports MC dropout - multiple voting_runs for val / test datasets
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._initialize_trainer()

    def _initialize_trainer(self):
        #init Hard_Mining_Status
        self._hard_mining_status = Hard_Mining_Status(19,1800)
        
        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = self.enable_cudnn

        if not self.has_training:
            self._cfg.training = self._cfg
            resume = bool(self._cfg.checkpoint_dir)
        else:
            resume = bool(self._cfg.training.checkpoint_dir)

        # Get device
        if self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
        self._device = torch.device(device)
        log.info("DEVICE : {}".format(self._device))

        # Profiling
        if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work well with it
            self._cfg.training.num_workers = 0

        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(self._cfg, self._cfg.wandb.public and self.wandb_log)

        # Checkpoint

        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_dir,
            self._cfg.model_name,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=resume,
        )

        # Create model and datasets
        if not self._checkpoint.is_empty:
            self._dataset: BaseDataset = instantiate_dataset(self._checkpoint.data_config)
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name=self._cfg.training.weight_name
            )
        else:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
            self._model.instantiate_optimizers(self._cfg)
            self._model.set_pretrained_weights()
            if not self._checkpoint.validate(self._dataset.used_properties):
                log.warning(
                    "The model will not be able to be used from pretrained weights without the corresponding dataset. Current properties are {}".format(
                        self._dataset.used_properties
                    )
                )
        self._checkpoint.dataset_properties = self._dataset.used_properties

        log.info(self._model)

        self._model.log_optimizers()
        log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Set dataloaders
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
            self._hard_mining_status
        )
        log.info(self._dataset)
        
        # Verify attributes in dataset
        self._model.verify_data(self._dataset.train_dataset[0])

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(self.wandb_log, self.tensorboard_log)

        if self.wandb_log:
            Wandb.launch(self._cfg, not self._cfg.wandb.public and self.wandb_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)
        if self.has_visualization:
            self._visualizer = Visualizer(
                self._cfg.visualization, self._dataset.num_batches, self._dataset.batch_size, os.getcwd()
            )

    def train(self):
        self._is_training = True
        
        for epoch in range(self._checkpoint.start_epoch, self._cfg.training.epochs):
            log.info("EPOCH %i / %i", epoch, self._cfg.training.epochs)
            
            
            self._test_epoch(epoch, "val")
            exit()
            self.update_hard_mining_status(epoch)
            #print(self._hard_mining_status)
            #exit()
            
            self._train_epoch(epoch)

            if self.profiling:
                return 0

            if epoch % self.eval_frequency != 0:
                continue

            if self._dataset.has_val_loader:
                self._test_epoch(epoch, "val")

            if self._dataset.has_test_loaders:
                self._test_epoch(epoch, "test")

        # Single test evaluation in resume case
        if self._checkpoint.start_epoch > self._cfg.training.epochs:
            if self._dataset.has_test_loaders:
                self._test_epoch(epoch, "test")

    def eval(self, stage_name=""):
        self._is_training = False

        epoch = self._checkpoint.start_epoch
        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                self._test_epoch(epoch, "val")

        if self._dataset.has_test_loaders:
            if not stage_name or stage_name == "test":
                self._test_epoch(epoch, "test")

    def _finalize_epoch(self, epoch):
        self._tracker.finalise(**self.tracker_options)
        if self._is_training:
            metrics = self._tracker.publish(epoch)
            self._checkpoint.save_best_models_under_current_metrics(self._model, metrics, self._tracker.metric_func)
            if self.wandb_log and self._cfg.wandb.public:
                Wandb.add_file(self._checkpoint.checkpoint_path)
            if self._tracker._stage == "train":
                log.info("Learning rate = %f" % self._model.learning_rate)

    def get_intersection_union_per_class(self, confusion_matrix):
    	TP_plus_FN = np.sum(confusion_matrix, axis=0)
    	TP_plus_FP = np.sum(confusion_matrix, axis=1)
    	TP = np.diagonal(confusion_matrix)
    	union = TP_plus_FN + TP_plus_FP - TP
    	iou = 1e-8 + TP / (union + 1e-8)
    	existing_class_mask = union > 1e-3
    	return iou, existing_class_mask

    def _count_confusion_matrix(self, ground_truth_vec, predicted):
    	number_of_labels = self._tracker._confusion_matrix.number_of_labels
    	assert np.max(predicted) < number_of_labels
    	if torch.is_tensor(ground_truth_vec):
    		ground_truth_vec = ground_truth_vec.numpy()
    	if torch.is_tensor(predicted):
    		predicted = predicted.numpy()    	
    	batch_confusion = np.bincount(number_of_labels * ground_truth_vec.astype(int) + predicted, minlength=number_of_labels ** 2).reshape(number_of_labels, number_of_labels)
    	
    	return batch_confusion
    
    def _get_cur_confusion_matrix(self):
    	outputs = self._model.get_output()
    	labels = self._model.get_labels()
    	mask = labels != self._tracker._ignore_label
    	outputs = outputs[mask]
    	labels = labels[mask]
    	outputs = self._tracker._convert(outputs)
    	labels = self._tracker._convert(labels)
    	predicts = np.argmax(outputs, 1)
    	batch_confusion = self._count_confusion_matrix(labels, predicts)
    	return batch_confusion
    
    def _train_epoch(self, epoch: int):

        self._model.train()
        self._tracker.reset("train")
        self._visualizer.reset(epoch, "train")
        train_loader = self._dataset.train_dataloader

        iter_data_time = time.time()
        with Ctq(train_loader) as tq_train_loader:
            for i, data in enumerate(tq_train_loader):
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                self._model.set_input(data, self._device)
                self._model.optimize_parameters(epoch, self._dataset.batch_size)
                if i % 10 == 0:
                    with torch.no_grad():
                        self._tracker.track(self._model, data=data, **self.tracker_options)

                tq_train_loader.set_postfix(
                    **self._tracker.get_metrics(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                    color=COLORS.TRAIN_COLOR
                )

                if self._visualizer.is_active:
                    self._visualizer.save_visuals(self._model.get_current_visuals())

                iter_data_time = time.time()

                if self.early_break:
                    break

                if self.profiling:
                    if i > self.num_batches:
                        return 0

        self._finalize_epoch(epoch)

    
    #This method is used to compute the property for hard mining
    def update_hard_mining_status(self, epoch):
        Instance_Conf_Mat = []
        Instance_Class_IoU = []
        Instance_Class_Mask = []
        
        
        self._tracker.reset("val")    	
        self._model.eval()
        if self.enable_dropout:
        	self._model.enable_dropout_in_eval()    	
        
        train_loader = self._dataset._hard_mining_loader
        iter_data_time = time.time()
        with Ctq(train_loader) as tq_train_loader:
        	for i, data in enumerate(tq_train_loader):
        		#if i > 50:
        		#	break
        		t_data = time.time() - iter_data_time
        		iter_start_time = time.time()
        		with torch.no_grad():
        			self._model.set_input(data, self._device)
        			self._model.forward(epoch=epoch)
        			self._tracker.track(self._model, data=data, **self.tracker_options)
        			
        			#Compute current instance confusion matrix
        			cur_confusion_matrix = self._get_cur_confusion_matrix()
        			#Compute current Class_IoU and Mask
        			Cur_Class_IoU, Cur_Class_Mask = self.get_intersection_union_per_class(cur_confusion_matrix)
        			Instance_Class_IoU.append(Cur_Class_IoU)
        			Instance_Class_Mask.append(Cur_Class_Mask)
        			Instance_Conf_Mat.append(cur_confusion_matrix)        			
        			
        			'''
        			Cur_Class_IoU, Cur_Class_Mask = self._tracker._confusion_matrix.get_intersection_union_per_class()
        			Instance_Class_IoU.append(Cur_Class_IoU)
        			Instance_Class_Mask.append(Cur_Class_Mask)
        			Instance_Conf_Mat.append(self._tracker._confusion_matrix.get_confusion_matrix())
        			print(self._tracker._confusion_matrix.get_confusion_matrix())
        			'''
        			        		
        #Instance Class IoU
        Instance_Class_IoU = np.array(Instance_Class_IoU)
        Instance_Conf_Mat = np.array(Instance_Conf_Mat)
        Instance_Class_Mask = np.array(Instance_Class_Mask).astype(int)
        
        
        #2 Class Wise IoU
        Class_IoU = np.sum(Instance_Class_IoU, axis = 0)
        Class_Mask_Cnt = np.sum(Instance_Class_Mask, axis = 0)
        Class_IoU = Class_IoU/Class_Mask_Cnt
        
        
        #3 Instance Wise IoU
        Instance_IoU = np.sum(Instance_Class_IoU, axis = 1)
        Instance_Mask_Cnt = np.sum(Instance_Class_Mask, axis = 1)
        Instance_IoU = Instance_IoU / Instance_Mask_Cnt
        
        Conf_Mat = np.sum(Instance_Conf_Mat, axis = 0) + 1
        
        #1 Confusion Matrix
        Conf_Mat = Conf_Mat / np.expand_dims(np.sum(Conf_Mat, axis=1), axis = 1)
        
        self._hard_mining_status.set(Conf_Mat,Class_IoU,Instance_IoU,Instance_Class_IoU,epoch)
                
        '''
        print(Conf_Mat.shape)
        print(Class_IoU.shape)
        print(Instance_IoU.shape)
        print(Instance_Class_IoU.shape)
        exit()
        '''
        	
        	

    def _test_epoch(self, epoch, stage_name: str):
        voting_runs = self._cfg.get("voting_runs", 1)
        if stage_name == "test":
            loaders = self._dataset.test_dataloaders
        else:
            loaders = [self._dataset.val_dataloader]

        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()

        for loader in loaders:
            stage_name = loader.dataset.name
            self._tracker.reset(stage_name)
            if self.has_visualization:
                self._visualizer.reset(epoch, stage_name)
            if not self._dataset.has_labels(stage_name) and not self.tracker_options.get(
                "make_submission", False
            ):  # No label, no submission -> do nothing
                log.warning("No forward will be run on dataset %s." % stage_name)
                continue

            for i in range(voting_runs):
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():
                            self._model.set_input(data, self._device)
                            self._model.forward(epoch=epoch)
                            self._tracker.track(self._model, data=data, **self.tracker_options)
                        tq_loader.set_postfix(**self._tracker.get_metrics(), color=COLORS.TEST_COLOR)

                        if self.has_visualization and self._visualizer.is_active:
                            self._visualizer.save_visuals(self._model.get_current_visuals())

                        if self.early_break:
                            break

                        if self.profiling:
                            if i > self.num_batches:
                                return 0

            test_confusion = self._tracker._confusion_matrix.get_confusion_matrix()
            label_number = np.sum(test_confusion, axis=1)
            predict_number = np.sum(test_confusion, axis=0)
            TP = np.diagonal(test_confusion)
            print(label_number)
            print("\n\n")
            print(predict_number)
            print("\n\n")
            print(TP)
            self._finalize_epoch(epoch)
            self._tracker.print_summary()

    @property
    def early_break(self):
        return getattr(self._cfg.debugging, "early_break", False) and self._is_training

    @property
    def profiling(self):
        return getattr(self._cfg.debugging, "profiling", False)

    @property
    def num_batches(self):
        return getattr(self._cfg.debugging, "num_batches", 50)

    @property
    def enable_cudnn(self):
        return getattr(self._cfg.training, "enable_cudnn", True)

    @property
    def enable_dropout(self):
        return getattr(self._cfg, "enable_dropout", True)

    @property
    def has_visualization(self):
        return getattr(self._cfg, "visualization", False)

    @property
    def has_tensorboard(self):
        return getattr(self._cfg, "tensorboard", False)

    @property
    def has_training(self):
        return getattr(self._cfg, "training", None)

    @property
    def precompute_multi_scale(self):
        return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)

    @property
    def wandb_log(self):
        if getattr(self._cfg, "wandb", False):
            return getattr(self._cfg.wandb, "log", False)
        else:
            return False

    @property
    def tensorboard_log(self):
        if self.has_tensorboard:
            return getattr(self._cfg.tensorboard, "log", False)
        else:
            return False

    @property
    def tracker_options(self):
        return self._cfg.get("tracker_options", {})

    @property
    def eval_frequency(self):
        return self._cfg.get("eval_frequency", 1)

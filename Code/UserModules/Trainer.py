# -*- coding: utf-8 -*-
"""
"""

import sklearn.metrics as sk_met
import numpy as np, pandas as pd
from collections import namedtuple
from ipywidgets import Output
from IPython.display import display_html, display
import os, json, torch
from torch.utils.data import DataLoader

pd.set_option('display.min_rows', 1000)
pd.set_option('display.precision', 3)

#### Define the Training Metrics class ####
Metric = namedtuple('Metric', 'name param_dict')      
class TrainingMetrics:
    def __init__(self, metrics2collect: list = [], metric_to_focus: str = 'loss', lowerIsbetter: bool = True):
        '''
        Parameters
        ----------
        metrics_2_collect : list of 'Metric' namedtuple.
            Lists of metrics to collect during training.
        metric_to_focus: str, optional
            Metric used to determine best model. The default is 'loss'.
        lowerIsbetter: bool, optional
            Is the objective metric being maximized or minimized. The default is True.
        '''
        datasets = ['training', 'development']
        # Unpack Metric functions
        self.metrics2collect = {m.name: {'func': getattr(sk_met, m.name), 'params': m.param_dict} for m in metrics2collect}

        # Create Metric Repositories
        self.metrics_collected = {d_t: {m.name: [] for m in metrics2collect} for d_t in datasets}
        # Add Loss
        self.metrics_collected['training']['loss'] = []
        self.metrics_collected['development']['loss'] = []

        if metric_to_focus == 'loss':
            assert lowerIsbetter, 'If metric optimized in validation is loss, then we expect a minimization problem.'
        self.lowerIsbetter = lowerIsbetter
            
        # Create Starting Value
        best_val = float('inf') if lowerIsbetter else float('-inf')
        self.best_validation = {'epoch': 0, 'step': 0, 'metric': metric_to_focus, 'value': best_val}
        
        # Set up display outputs
        self.output = Output()
        display(self.output)
        
    def update_metrics(self, data_type, y_true, y_pred, mean_loss, epoch, step):
        '''
        Computes the metrics based on the results of the run
        Parameters
        ----------
        data_type : str
            denotes the source of the results. Must be in {'training', 'development'}.
        y_true : list
            True Label results in the run.
        y_pred : list
            Predicted Label results in the run.
        mean_loss: float
            Average Loss for the run
        epoch : float
            Epoch number.
        step : int
            Step number.
        '''
        # Compute average loss
        self.metrics_collected[data_type]['loss'].append({'epoch': epoch, 'step': step, 'value': mean_loss})
        
        # Compute other metrics
        for metric_key, m_dict in self.metrics2collect.items():
            value = m_dict['func'](y_true, y_pred, **m_dict['params']) # Compute metric
            self.metrics_collected[data_type][metric_key].append({'epoch': epoch, 'step': step, 'value': value})
            
    def update_best_validation(self):
        '''
        Check whether best validation score was improved in the latest evaluation iteration.
        Returns
        -------
        improved_validation : bool
            Boolean indicating if the best validation score was updated.

        '''
        best_met_name = self.best_validation['metric']
        
        # Check if a better validation score was attained in the previous iteration
        last_val =  self.metrics_collected['development'][best_met_name][-1]
        validation_improved = last_val['value'] <  self.best_validation['value'] if self.lowerIsbetter else last_val['value'] >  self.best_validation['value']
        
        if validation_improved: # Update Local Optima
            self.best_validation = {'epoch': last_val['epoch'], 'step': last_val['step'], 'metric': best_met_name, 'value': last_val['value']}
            
        return validation_improved
    
        
    def save(self, outDir, save_result_csv = True):
        
        outJSON = {
                'metrics2collect': [{'name': metric_key, 'params': m_dict['params']} for metric_key, m_dict in self.metrics2collect.items()],
                'metrics_collected': self.metrics_collected,
                'lowerIsbetter': self.lowerIsbetter,
                'best_validation': self.best_validation
        }
        with open(os.path.join(outDir, 'TrainingMetrics.json'), 'w+',encoding='utf8') as outF:
            outF.write(json.dumps(outJSON, indent=4))      
                   
        if save_result_csv:
            self.get_results_dataframe()
            self.resultsDF.to_csv(os.path.join(outDir, 'metrics.csv'), index = False)

    def get_results_dataframe(self):
        elem = self.metrics_collected['training']['loss'][0]
        for i, (dataType, dataDict) in enumerate(self.metrics_collected.items()):
            reformed = {(outerKey, key): [d[key] for d in innerList] for key in elem for outerKey, innerList in dataDict.items()}
            temp = pd.DataFrame(reformed).stack(0).reset_index(1).pivot(index = ['epoch', 'step'], columns = 'level_1', values = 'value').reset_index()
            temp['data_source'] = dataType
            if i == 0:
                self.resultsDF = temp.copy()
            else:
                self.resultsDF = self.resultsDF.append(temp)
        self.print_results()        
        
    def print_results(self):
        with self.output:
            self.output.clear_output(wait = True)
            # Prepare Training output
            trainTemp = self.resultsDF[self.resultsDF.data_source == 'training'].drop(columns = 'data_source')
            train_styler = trainTemp.style.set_table_attributes("style='display:inline'").set_caption('Training')
            # Prepare Development output
            devTemp = self.resultsDF[self.resultsDF.data_source == 'development'].drop(columns = 'data_source')
            dev_styler = devTemp.style.set_table_attributes("style='display:inline'").set_caption('Development')

            space = "\xa0" * 10
            display_html(train_styler._repr_html_() + space  + dev_styler._repr_html_(), raw=True)
            
    def load(self, results_path):
        # Load the JSON of the class
        with open(os.path.join(results_path, 'TrainingMetrics.json'), 'r',encoding='utf8') as inF:
            inJSON = json.load(inF)
        
        # Instantiate the class
        self.metrics2collect = {m['name']: {'func': getattr(sk_met, m['name']), 'params': m['params']} for m in inJSON['metrics2collect']}
        self.metrics_collected = inJSON['metrics_collected']
        self.lowerIsbetter = inJSON['lowerIsbetter']
        self.best_validation = inJSON['best_validation']
                       
        # Print Results
        self.get_results_dataframe()
        
#### Define the Trainer class ####
import shutil, random
from dataclasses import dataclass, field

@dataclass
class TrainerConfig:
    """
    Configuration for the model used to encode a user's tweets.
    """
    # Required Fields
    epochs_to_train: int =  field(
        metadata={"help": "Number of epochs to train the model."}
    )
    results_path: str =  field(
        metadata={"help": "Directory used to save checkpoints and best models."}
    )
    optimizer: torch.optim.Optimizer =  field(
        metadata={"help": "Optimizer used for training."}
    )
    batch_size: int =  field(
        metadata={"help": "Batch size used for training."}
    )
    train_dataloader: DataLoader =  field(
        metadata={"help": "Iterator used for training the model."}
    )
    val_dataloader: DataLoader =  field(
        metadata={"help": "Iterator used for validating model performance."}
    )   
    
    # Other Trainer Info
    device: str =  field(
        default = 'cpu', metadata={"help": "Device used for training. Defaults to CPU."}
    )        
    loss: torch.nn.Module =  field(
        default = torch.nn.CrossEntropyLoss(), metadata={"help": "Loss function used for training."}
    )    
    scheduler: torch.optim.lr_scheduler =  field(
        default = None, metadata={"help": "Learning Rate scheduler used during training. Optional."}
    )
    clip_max_norm: float =  field(
        default = None, metadata={"help": "List of model arguments names. Must also match the names specified by the dataloaders provided. If these are not provided, then all elements of the batch are passed to the model in order."}
    )
    steps_to_eval: int =  field(
        default = None, metadata={"help": "Steps needed before evaluation step. If not provided, it defaults to the number of steps in one epoch"}
    )    
    early_stop_eval_steps: float =  field(
        default = float('inf'), metadata={"help": "Maximum number of evaluation steps without improvement before stopping training. Optional."}
    )
    max_checkpoints_num: float =  field(
        default = float('inf'), metadata={"help": "Maximum number of evaluation steps without improvement before stopping training. Optional."}
    ) 
    seed: int =  field(
        default = None, metadata={"help": "Seed used for reproducibility of results. Optional."}
    )
    use_notebook: bool =  field(
        default = True, metadata={"help": "Whether the training is taking place on a Jupyter Notebook. Defaults to True."}
    )
    
    # Other Iterator Info
    x_names: list =  field(
        default_factory = [], metadata={"help": "List of model arguments names. Must also match the names specified by the dataloaders provided. If these are not provided, then all elements of the batch are passed to the model in order."}
    )
    y_name: str =  field(
        default = None, metadata={"help": "Name of the label variable as defined by the dataloader. If not provided, then its assumed to be the last element in the batch."}
    )    
    resume_checkpoint_path: str =  field(
        default = None, metadata={"help": "Path were to load trained checkpoints. If not None, then training will resume from specified checkpoint."}
    )
 
    # TrainerMetrics parameters
    metrics2collect: list =  field(
        default_factory = [], metadata={"help": "List of 'Metric' namedtuples."}
    )
    metric_to_focus: str =  field(
        default = 'loss', metadata={"help": "Metric used to determine best model. The default is 'loss'."}
    )
    lowerIsbetter: bool =  field(
        default = True, metadata={"help": "Is the objective metric being maximized or minimized. The default is True."}
    )
    
    
    def __post_init__(self):
        if self.steps_to_eval is None:  self.steps_to_eval = len(self.train_iterator)
        self.epochs_to_train = int(self.epochs_to_train)

        # Prepare Out Directories
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)
        self.best_model_dir = os.path.join(self.results_path, 'Best_Models')
        if not os.path.isdir(self.best_model_dir):
            os.mkdir(self.best_model_dir)            
        
        # Check if Training Metrics where passed correctly
        assert isinstance(self.metrics2collect[0], Metric), 'metrics2collect must be an iterable of Metric named tuples.'
        
        # Validate Model Arguments
        if len(self.x_names) > 0:
            if isinstance(self.x_names, str):   self.x_names = [self.x_names]
            if isinstance(self.x_names, list):
                assert self.y_name is not None, 'If names are assigned to model arguments, then a name must be provided for the predicted variable (label)'
        else:
            assert self.y_name is None, 'As model argument names were not provided, no name for the predicted variable should be given.'
        
        

from tqdm.notebook import tqdm as tqdm_note, trange as trange_note
from tqdm import tqdm, trange

class Trainer:
    def __init__(self, model, config: TrainerConfig):
        
        self.model = model
        self.model.zero_grad()
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.loss = config.loss
        self.config = config
        
        # Create Containers for checkpoints and best models
        self.saved_checkpoints = []
        self.saved_best_models = []
        
        # Load Checkpoint if one is provided
        if config.resume_checkpoint_path is not None:
            self.epochs_trained, self.steps_trained, self.global_step = self.load_checkpoint()
        else:            
            self.epochs_trained, self.steps_trained, self.global_step =  0, 0, 0
            # Create Container for collected metrics
            self.training_metrics = TrainingMetrics(config.metrics2collect, config.metric_to_focus, config.lowerIsbetter)
            
        # Get training and evaluation observations
        self.train_obs, self.validation_obs = config.batch_size * len(config.train_dataloader), config.batch_size * len(config.val_dataloader)
        
        # Set Appropriate Counters
        self.trange = trange_note if self.config.use_notebook else trange
        self.tqdm = tqdm_note if self.config.use_notebook else tqdm
            
        if config.seed is not None:
            self.set_seed()
        
    def save_checkpoint(self, save_path, valid_loss, epoch, step, total_steps):
        checkDir = os.path.join(save_path, 'checkpoint_{}'.format(total_steps)) 
        if not os.path.isdir(checkDir):
            os.mkdir(checkDir)            
        
        fName = os.path.join(checkDir, 'model_results.pth')
    
        state_dict = {'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict':  self.optimizer.state_dict(),
                      'valid_loss': valid_loss,
                      'epoch': epoch,
                      'step_in_epoch': step,
                      'total_steps': total_steps}
        
        if self.scheduler is not None:
            state_dict['scheduler_state_dict'] = self.scheduler.state_dict()    
        
        torch.save(state_dict, fName)
        print(f'    Model saved to ==> {checkDir}')
        
        # Save Training Metrics
        self.training_metrics.save(checkDir, save_result_csv = True)
        return checkDir
    
    def load_checkpoint(self):
        fName = os.path.join(self.config.resume_checkpoint_path, 'model_results.pth')
        state_dict = torch.load(fName, map_location = self.config.device)
        print(f'Model loaded from <== {self.config.resume_checkpoint_path}')
        
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        
        # Load Training Metrics
        self.training_metrics = TrainingMetrics()
        self.training_metrics.load(self.config.resume_checkpoint_path)            
        
        return state_dict['epoch'], state_dict['step_in_epoch'], state_dict['total_steps']
    
    def set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        
    def train(self):
        self.model = self.model.train()

        eval_steps_without_improvement = 0
        stop_training = False
        
        train_iterator = self.trange(self.epochs_trained, self.config.epochs_to_train, desc="Epoch")
            
        for epoch in train_iterator:
            ## These should reset every epoch
            if stop_training:    break
            running_loss = 0.0
            train_lbl_pred, train_lbl_true = [], [] 
            epoch_iterator = self.tqdm(self.config.train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                if step <= self.steps_trained:
                    continue # Skip to last step trained
                    
                # Get Batch Elements
                if len(self.config.x_names) > 0:
                    X = {x_n: batch[x_n].to(self.config.device) for x_n in self.config.x_names}
                    labels = batch[self.config.y_name].to(self.config.device) 
                    # Predict
                    model_output = self.model(**X)
                else: # When argument names are not provided we asume the label is the last element in the batch tuple
                    X = (x.to(self.config.device) for x in batch[:-1])
                    labels = batch[-1].to(self.config.device) 
                    model_output = self.model(*X)
                # Resolve output
                if isinstance(model_output, tuple): # If model outputs a tuple we will take the first element
                    logits = model_output[0]
                else: # We assume that the output only outputs one element
                    logits = model_output
                          
                _, preds = torch.max(logits, dim=1)
        
                loss_val = self.loss(logits, labels)
        
                # Do backward pass
                self.optimizer.zero_grad()
                loss_val.backward()
                #plot_grad_flow(model.named_parameters())
                self.optimizer.step()
                if self.scheduler is not None:
                    # Update the learning rate.
                    self.scheduler.step()   
                if self.config.clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.config.clip_max_norm)
                # Update Metrics
                self.global_step += 1
                running_loss += loss_val.item()
                train_lbl_true.extend(list(labels.cpu().numpy()))
                train_lbl_pred.extend(list(preds.cpu().numpy()))   
        
                # evaluation step
                if self.global_step % self.config.steps_to_eval == 0:
                    # Get Training Results
                    self.training_metrics.update_metrics('training', train_lbl_true, train_lbl_pred, running_loss / len(train_lbl_true), epoch, self.global_step)
        
                    # Get Validation Results
                    valid_true, valid_pred, valid_loss = self.evaluate()
                    self.training_metrics.update_metrics('development', valid_true, valid_pred, valid_loss, epoch, self.global_step)
        
                    # Update optima
                    validation_improved = self.training_metrics.update_best_validation()
                    self.model.train()
         
                    # Save Model and Metrics
                    saved_dir = self.save_checkpoint(self.config.results_path, valid_loss, epoch, step, self.global_step)
                    self.saved_checkpoints.append(saved_dir)
                    
                    if len(self.saved_checkpoints) > self.config.max_checkpoints_num:
                        dir2remove = self.saved_checkpoints.pop(0)
                        shutil.rmtree(dir2remove)            
                    
                    if validation_improved: # Save if best evaluation is attained
                        best_valid_loss = self.training_metrics.best_validation['value']
                        eval_steps_without_improvement = 0
                        # Save Best Model In dedicated directory
                        saved_dir = self.save_checkpoint(self.config.best_model_dir, best_valid_loss, epoch, step, self.global_step)
                        self.saved_best_models.append(saved_dir)                   
                        
                        # Delete if more than 2 best models have been saved
                        if len(self.saved_best_models) > 2:
                            dir2remove = self.saved_best_models.pop(0)
                            shutil.rmtree(dir2remove)   
                        
                    else:
                        eval_steps_without_improvement += 1
                        if eval_steps_without_improvement == self.config.early_stop_eval_steps:
                            stop_training = True
                            break
            if self.steps_trained > 0:  self.steps_trained = 0 # Reset steps trained to avoid next epoch to also skip those steps.
        self.epochs_trained, self.steps_trained = epoch, step

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():   
            lbl_true, lbl_pred = [], []        
            valid_running_loss = 0.0                 
            # validation loop
            epoch_iterator = self.tqdm(self.config.val_dataloader, desc="Evaluation")
            for step, batch in enumerate(epoch_iterator):      
                # Get Batch Elements
                if len(self.config.x_names) > 0:
                    X = {x_n: batch[x_n].to(self.config.device) for x_n in self.config.x_names}
                    labels = batch[self.config.y_name].to(self.config.device) 
                    # Predict
                    model_output = self.model(**X)
                else: # When argument names are not provided we asume the label is the last element in the batch tuple
                    X = (x.to(self.config.device) for x in batch[:-1])
                    labels = batch[-1].to(self.config.device) 
                    model_output = self.model(*X)
                # Resolve output
                if isinstance(model_output, tuple): # If model outputs a tuple we will take the first element
                    logits = model_output[0]
                else: # We assume that the output only outputs one element
                    logits = model_output                
                _, preds = torch.max(logits, dim=1)
                
                loss_val = self.loss(logits, labels)
                valid_running_loss += loss_val.item()  
                # Get Stats for the run
                lbl_true.extend(list(labels.data.cpu().numpy()))
                lbl_pred.extend(list(preds.cpu().numpy()))
        return lbl_true, lbl_pred, valid_running_loss / len(lbl_pred)
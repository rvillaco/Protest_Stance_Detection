import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from config import *
from homo_import_balanced import gen_import_dict, import_data, self_loader
from sklearn.metrics import  accuracy_score, f1_score
from homo_model import dumb_GNN, dumbest_GNN
import random

# for balanced, lets make it a bit random
#torch.manual_seed(1234)
#random.seed(1234)
#np.random.seed(1234)


# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# debug time..
#device = torch.device('gpu')

def init_params():
    with torch.no_grad():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(device)
        model(batch.x, batch.edge_index)

def train():
    """
    Trains the model
    """
    model.train()
    total_loss = total_correct = 0
    train_loader_len = 0
    possible_correct = 0
    predictions = []
    true_labs = []
    
    for batch in tqdm(train_loader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        # if the batch contains no train nodes, loss becomes nan and gets overridden.
        # we'll only update if at least one of our target nodes is in the training set

        # if loss is ever nan, we can add the commented out if/else statement below
        # we can't have a batch of all unlabeled nodes. This shouldn't be an issue
        # with how the train loader is created

        # import the training mask- we'll use this to ensure only nodes with training labels get counted when

        batch_size = batch.batch_size
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)[:batch_size]
        loss = loss_fn(out, batch.y[:batch_size])
        
        loss.backward()
        optimizer.step()
        
        pred = out.argmax(dim=-1)
        
        possible_correct += batch_size
        total_correct += int((pred == batch.y[:batch_size]).sum())
        
        total_loss += float(loss)
        train_loader_len += 1
        
        predictions.append(pred.cpu().numpy())
        true_labs.append(batch.y[:batch_size].cpu().numpy())
        

    train_loss = total_loss / train_loader_len
    #train_acc = total_correct / possible_correct
    train_acc = accuracy_score(np.concatenate(true_labs), np.concatenate(predictions))
    f1 = f1_score(np.concatenate(true_labs), np.concatenate(predictions), average = 'macro')

    return train_loss, train_acc, f1

def test(loader):
    with torch.no_grad():
        model.eval()
        losses = []
        total_loss = total_correct = 0
        train_loader_len = 0
        possible_correct = 0
        total_examples = total_correct = 0
        predictions = []
        true_labs = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            batch_size = batch.batch_size
            out = model(batch.x, batch.edge_index)[:batch_size]
            pred = out.argmax(dim=-1)

            # calculate validation loss
            loss = loss_fn(out, batch.y[:batch_size])

            pred = out.argmax(dim=-1)
            
            #possible_correct += batch_size
            #total_correct += int((pred == batch.y[:batch_size]).sum())
            
            total_loss += float(loss)
            train_loader_len += 1
            
            predictions.append(pred.cpu().numpy())
            true_labs.append(batch.y[:batch_size].cpu().numpy())
        

            
    valid_loss = total_loss / train_loader_len
    #valid_acc = total_correct / possible_correct           
    valid_acc = accuracy_score(np.concatenate(true_labs), np.concatenate(predictions))
    f1 = f1_score(np.concatenate(true_labs), np.concatenate(predictions), average = 'macro')

    return valid_loss, valid_acc, f1


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
        not improving
        :param min_delta: minimum difference between new loss and old loss for
        new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_model = None
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss, model):
        if self.best_loss == None:
            self.best_loss = val_loss
            #self.best_model = copy.deepcopy(model)
            self.best_model = model

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            #self.best_model = copy.deepcopy(model)
            self.best_model = model
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


if __name__ == '__main__':
    ## Run the script
    for country_name in ['Ecuador', 'Bolivia', 'Colombia', 'Chile']:
        early_stopping = EarlyStopping()
        training_stats = []
        import_dict = gen_import_dict(country_name)
        att_df, idx_df, mask_and_y_cache, true_y = import_data(import_dict, balanced_data=False)
        #instantiate on the 0th epoch- we need data.metadata to convert the model to heterogeneous
        # we can use these loaders when we run the first epoch
        train_loader, valid_loader, test_loader = self_loader(att_df, idx_df, mask_and_y_cache, true_y)
        
        #lets free up some mem:
        if country_name == 'Chile':
            del att_df
            del idx_df
        
        model = dumb_GNN(512, 2).to(device)
        #model = dumbest_GNN(2).to(device)
        #model = to_hetero(model, data.metadata(), aggr='sum').to(device)
        init_params()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(0, EPOCHS):
            #if epoch > 0:
                # chile too big for memory- kludgey work-around
                #if country_name == 'Chile':
                #    del train_loader
                #    del valid_loader
                #    del test_loader
                #    del data
                #    att_df, idx_df, mask_and_y_cache, true_y = import_data(import_dict)
                #    data, train_loader, valid_loader, test_loader = balanced_loader(att_df, idx_df, mask_and_y_cache, true_y, epoch)
                #    del att_df
                #    del idx_df
            #    if HOMO:
            #        pass
            #    else:
                    #data, train_loader, valid_loader, test_loader = balanced_loader(att_df, idx_df, mask_and_y_cache, true_y, epoch)
            tloss, tacc, tf1 = train()
            #print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            vloss, vacc, vf1 = test(valid_loader)
            _, test_accuracy, test_f1 = test(test_loader)
            early_stopping(vloss, model)
            if early_stopping.early_stop:
                break
        print(f'Epoch: {epoch:02d}, Train Loss: {tloss:.4f}, Train Acc: {tacc:.4f}\
                Val Acc: {vacc:.4f}, Valid Loss: {vloss:.4f}')
        torch.save(early_stopping.best_model, f'../models/{country_name}_bal_gnn.pt')
        del model


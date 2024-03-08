import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from src.utils import *
from src.models import load_model
from src.utils import args
import optuna


def valid_step(model, criterion, val_loader):
    model.eval()
    soft = nn.Softmax(dim=1)
    avg_loss = 0.0
    avg_acc = 0.0
    y_pred = []
    prediction = []
    y_label = []
    with torch.no_grad():
        for (sample_1, labels_1), (sample_2, labels_2), (sample_3, lables_3) in zip(val_loader['omic1'], val_loader['omic2'], val_loader['omic3']):
            # forward pass
            sample_1, labels_1, sample_2, sample_3 = sample_1.to(args.device), labels_1.to(args.device), sample_2.to(args.device), sample_3.to(args.device)
            
            outputs = model(sample_1, sample_2, sample_3)

            # if needed when the batch size is 1
            loss = criterion(outputs, labels_1.squeeze(dim=1))

            #y_pred += outputs.squeeze().tolist()
            prediction += (soft(outputs)).squeeze(dim=1).tolist()
            y_label += labels_1.squeeze(dim=1).tolist()
           
            # gather statistics
            avg_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            y_pred += preds.tolist()
            avg_acc += torch.sum(preds == labels_1.squeeze(dim=1)).item()

    
    if args.dataset == 'ROSMAP':
        pred_proba = np.array(prediction)[:,1]
        auc = roc_auc_score(np.array(y_label), pred_proba)
        f1 = f1_score(np.array(y_label), np.array(y_pred))

        return {'loss': avg_loss / len(val_loader['omic1']), 'accuracy': avg_acc / len(y_label), 'auc': auc, 'labels': y_label, 'pred': y_pred, 'f1':f1}
    
    else:
        f1_mc = f1_score(np.array(y_label), np.array(y_pred), average='macro')
        f1_we = f1_score(np.array(y_label), np.array(y_pred), average='weighted')

        return {'loss': avg_loss / len(val_loader['omic1']), 'accuracy': avg_acc / len(y_label), 'labels': y_label, 'pred': y_pred, 'f1':[f1_mc,f1_we]}



def train_step(model, criterion, optimizer, train_loader):
    model.train()
    avg_loss = 0.0
    avg_acc = 0.0
    y_label = 0
    for (sample_1, labels_1), (sample_2, labels_2), (sample_3, lables_3) in zip(train_loader['omic1'], train_loader['omic2'], train_loader['omic3']):
        optimizer.zero_grad()
        # forward pass
        sample_1, labels_1, sample_2, sample_3 = sample_1.to(args.device), labels_1.to(args.device), sample_2.to(args.device), sample_3.to(args.device)
        y_label += labels_1.shape[0]

        # prediction
        probs = model(sample_1, sample_2, sample_3)
        #print(probs)
        # loss
        loss = criterion(probs, labels_1.squeeze(dim=1))
        #print(labels_1.squeeze(dim=1))
        # back-prop
        loss.backward()
        optimizer.step()
        
        # gather statistics
        avg_loss += loss.item()
        #print(f'loss item {loss.item()}')
        _, preds = torch.max(probs, 1)
        avg_acc += torch.sum(preds == labels_1.squeeze(dim=1)).item()

    return {'loss': avg_loss / len(train_loader['omic1']), 'accuracy': avg_acc / y_label}


def train_model(trial, seed):
    fix_random_seed(seed=seed)
    # define search space for tuning hyperparameters
    if args.dataset == "ROSMAP":
        min_principal = 180
        max_principal = 200
        params = { 
                'learning_rate':trial.suggest_categorical('learning_rate', [5e-5]),
                'weight_decay': trial.suggest_categorical('weight_decay',  [1e-4]),
                'optimizer': trial.suggest_categorical("optimizer", ['Adam']),
                'batch_size': trial.suggest_categorical('batch_size', [32]),
                'dropout1': trial.suggest_categorical('dropout1', [0.5]),
                'kernel_par_1': trial.suggest_categorical('kernel_par_1', [0.0005, 0.0007, 0.001]), #[0.0005, 0.0001, 0.00005] [0.0005, 0.0007, 0.001]
                'kernel_par_2': trial.suggest_categorical('kernel_par_2', [0.0005, 0.0007, 0.001]),
                'kernel_par_3': trial.suggest_categorical('kernel_par_3', [0.0005, 0.0007, 0.001]), 
                'n_principal': trial.suggest_categorical('n_principal', [120]),
                'epochs': trial.suggest_int('epochs', 120, 200, step=10)
                }
    else:
        
        min_principal = 9
        max_principal = 18
    
        params = { 
                'learning_rate':trial.suggest_categorical('learning_rate', [1e-4]),
                'weight_decay': trial.suggest_categorical('weight_decay',  [0]),
                'optimizer': trial.suggest_categorical("optimizer", ['Adam']),
                'dropout1': trial.suggest_categorical('dropout1', [0.3]),
                'batch_size': trial.suggest_categorical('batch_size', [32]),
                'kernel_par_1': trial.suggest_categorical('kernel_par_1', [0.00005,0.0005,0.005]), 
                'kernel_par_2': trial.suggest_categorical('kernel_par_2', [0.00005,0.0005,0.005]),
                'kernel_par_3': trial.suggest_categorical('kernel_par_3', [0.00005,0.0005,0.005]),
                'n_principal': trial.suggest_int('n_principal', min_principal, max_principal, step=3),
                'epochs': trial.suggest_int('epochs', 120, 200, step=10)
                }
    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if 'best_epoch' in t.params.keys():
            del t.params["best_epoch"]
        if t.params == trial.params:
            raise optuna.exceptions.TrialPruned('Duplicate parameter set')
        


    data_1, data_2, data_3 = load_data(seed=seed, kernel_par = [params['kernel_par_1'],params['kernel_par_2'], params['kernel_par_3']],
                                       n_principal= params['n_principal'])
    #print(data_1[0][0])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    n_splits = 5
    # define splits for the k-fold
    splits= StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    # scores of a configuration of hyperparameters on each fold
    scores_tr = []
    scores_val = []
    #scores_per_epoch = {'train': np.zeros(n_epoch), 'val': np.zeros(n_epoch)}
    for fold, (train_idx, val_idx) in enumerate(splits.split(data_1[0][:][0], data_1[0][:][1])):
        fix_random_seed(seed=seed)
        input_size_1, input_size_2, input_size_3 = data_1[0][0][0].shape[0], data_2[0][0][0].shape[0], data_3[0][0][0].shape[0]
        model = load_model(params, input_size_1, input_size_2, input_size_3, args.model).to(args.device)

        # sampling from omic1 (same observations for each omic)
        np.random.shuffle(train_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        #print(list(train_sampler))
        valid_sampler = val_idx
        kwargs = {} if args.device=='cpu' else {'num_workers': 2, 'pin_memory': True}
        loader_kwargs = {**kwargs}

        # omic1
        train_loader_1 = DataLoader(data_1[0], **loader_kwargs, sampler= train_sampler, batch_size= params['batch_size'])
        val_loader_1  = DataLoader(data_1[0], **loader_kwargs, sampler= valid_sampler, batch_size=params['batch_size'])  

        # omic2
        train_loader_2 = DataLoader(data_2[0], **loader_kwargs, sampler=train_sampler, batch_size=params['batch_size'])
        val_loader_2  = DataLoader(data_2[0], **loader_kwargs, sampler= valid_sampler, batch_size=params['batch_size']) 

        # omic3
        train_loader_3 = DataLoader(data_3[0], **loader_kwargs, sampler=train_sampler, batch_size=params['batch_size'])
        val_loader_3  = DataLoader(data_3[0], **loader_kwargs, sampler= valid_sampler, batch_size=params['batch_size']) 

        # loader for each omic, train and validation
        train_loader =  {'omic1': train_loader_1, 'omic2': train_loader_2, 'omic3': train_loader_3}
        val_loader =  {'omic1': val_loader_1, 'omic2': val_loader_2, 'omic3': val_loader_3}

        # define optimizer
        if params['optimizer'] in ["Adagrad", "Adam", "AdamW", "SGD"]:
            optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
        else:
            optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])

        for epoch in range(params['epochs']):
            train_stats = train_step(model, criterion, optimizer, train_loader)
        valid_stats = valid_step(model, criterion, val_loader)

        scores_tr.append(train_stats['accuracy'])
        scores_val.append(valid_stats['accuracy'])

    print(f'Train acc mean over folds: {np.mean(scores_tr)}, Train acc std {np.std(scores_tr)}')
    print(f'Val acc mean over folds: {np.mean(scores_val)}, Val acc std {np.std(scores_val)}')

    # average of metric on the k folds for 1 set of hyperparameters
    return np.mean(scores_val)


if __name__ == "__main__":
    pass

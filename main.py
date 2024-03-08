from torch.utils.data import SubsetRandomSampler, DataLoader
from src.utils import args
from src.models import load_model
from src.utils.load_data import load_data
from src.utils import *
from src.opt.training_pipeline import train_model, train_step, valid_step
import optuna
import torch.optim as optim
import pandas as pd


def single_run(seed):
    fix_random_seed(seed)

    objective = lambda trial: train_model(trial, seed)

    # RUN loop for hyper-parameter optimization
    sampler = optuna.samplers.TPESampler(seed=1)
    study = optuna.create_study(direction="maximize", sampler=sampler) # accuracy as metric to optimize parameters
    unique_trials = 20
    while unique_trials > len(set(str(t.params) for t in study.trials)):
        study.optimize(objective, n_trials=1)

    # best set of hyperparameter selected
    best_params = study.best_trial.params

    kwargs = {} if args.device=='cpu' else {'num_workers': 2, 'pin_memory': True}
    loader_kwargs = {**kwargs}

    data_1, data_2, data_3 = load_data(seed=seed, kernel_par = [best_params['kernel_par_1'], best_params['kernel_par_2'], best_params['kernel_par_3']],
                                        n_principal= best_params['n_principal'])
    #print(data_1[0][0])

    # Create indices for the dataset that you want to shuffle
    fix_random_seed(seed)
    indices_view1 = np.arange(len(data_1[0]))
    np.random.shuffle(indices_view1)


    # Create SubsetRandomSampler using the shuffled indices
    sampler_view1 = SubsetRandomSampler(indices_view1)

    # omic1
    train_loader_1 = DataLoader(data_1[0], **loader_kwargs, sampler=sampler_view1, batch_size=best_params['batch_size'])
    test_loader_1  = DataLoader(data_1[1], **loader_kwargs, batch_size=best_params['batch_size'])  
    # omic2
    train_loader_2 = DataLoader(data_2[0], **loader_kwargs, sampler=sampler_view1, batch_size=best_params['batch_size'])
    test_loader_2  = DataLoader(data_2[1], **loader_kwargs, batch_size=best_params['batch_size'])  
    # omic3
    train_loader_3 = DataLoader(data_3[0], **loader_kwargs, sampler=sampler_view1, batch_size=best_params['batch_size'])
    test_loader_3  = DataLoader(data_3[1], **loader_kwargs, batch_size=best_params['batch_size']) 

    # loader for each omic, train and validation
    train_loader =  {'omic1': train_loader_1, 'omic2': train_loader_2, 'omic3': train_loader_3}
    test_loader =  {'omic1': test_loader_1, 'omic2': test_loader_2, 'omic3': test_loader_3}

    input_size_1, input_size_2, input_size_3 =  data_1[0][0][0].shape[0], data_2[0][0][0].shape[0], data_3[0][0][0].shape[0]

    # initialize besat model
    best_model = load_model(best_params, input_size_1, input_size_2, input_size_3, args.model).to(args.device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    if best_params['optimizer'] in  ["Adagrad", "Adam", "AdamW", "SGD"]:
        best_optimizer = getattr(optim, best_params['optimizer'])(best_model.parameters(), lr= best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    else:
        best_optimizer = getattr(optim, best_params['optimizer'])(best_model.parameters(), lr= best_params['learning_rate'])

    for epoch in range(best_params['epochs']):
        train_stats = train_step(best_model, criterion, best_optimizer, train_loader)

    test_stats = valid_step(best_model, criterion, test_loader)

    print(f'Best hyperparameters for this split: {best_params}')
    print(f'Train Accuracy: {train_stats["accuracy"]}')
    print(f'Test Accuracy: {test_stats["accuracy"]}')
    print()
    if args.dataset == 'ROSMAP':
        print(f'Test AUC: {test_stats["auc"]}')
        print(f'Test F1: {test_stats["f1"]}')
        return test_stats["accuracy"], test_stats["auc"], test_stats["f1"]

    else:
        print(f'Test F1_macro: {test_stats["f1"][0]}')
        print(f'Test F1_weighted: {test_stats["f1"][1]}')
        return test_stats["accuracy"], test_stats["f1"][0], test_stats['f1'][1]



def main():
    accuracies = []
    aucs = []
    f1s = []
    f1_mcs = []
    f1_wes = []
    if args.dataset=='ROSMAP':
        for seed in range(5):
            accuracy, auc, f1 = single_run(seed=seed)
            accuracies.append(accuracy)
            aucs.append(auc)
            f1s.append(f1)

        print(f'Average TEST ACC over 5 runs: {np.mean(accuracies)}, STD {np.std(accuracies)}')
        print(f'Average TEST AUC over 5 runs: {np.mean(aucs)}, STD {np.std(aucs)}')
        print(f'Average TEST F1 over 5 runs: {np.mean(f1s)}, STD {np.std(f1s)}')
    else:
        for seed in range(5):
            accuracy, f1_mc, f1_we = single_run(seed=seed)
            accuracies.append(accuracy)
            f1_mcs.append(f1_mc)
            f1_wes.append(f1_we)

        print(f'Average TEST ACC over 5 runs: {np.mean(accuracies)}, STD {np.std(accuracies)}')
        print(f'Average TEST F1 macro over 5 runs: {np.mean(f1_mcs)}, STD {np.std(f1_mcs)}')
        print(f'Average TEST F1 Weightew 5 runs: {np.mean(f1_wes)}, STD {np.std(f1_wes)}')

        
    

if __name__ == "__main__":
    main()
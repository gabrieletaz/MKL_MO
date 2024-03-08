import torch
import argparse


# ----- Parser -----

def parser():
    PARSER = argparse.ArgumentParser(description='Training parameters.')

    PARSER.add_argument('--dataset', default='ROSMAP', type=str,
                        choices=['ROSMAP', 'BRCA'],
                        help='Dataset.')
    
    PARSER.add_argument('--model', default='NN_sum', type=str,
                        choices=['NN_concat','CrossModal_NN_concat'], 
                        help='Model.')

    PARSER.add_argument('--device', default='cpu', type=str,
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:3', 'cuda:2'],
                        help='Device to run the experiment.')

    PARSER.add_argument('--exp_name', default='test', type=str)
    
    ARGS = PARSER.parse_args()

    if ARGS.device is None:
        ARGS.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #print_args(ARGS)

    return ARGS

args = parser()

if __name__ == "__main__":
    pass

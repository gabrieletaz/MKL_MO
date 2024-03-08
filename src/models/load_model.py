import torch.nn as nn
import torch
from src.utils import args


# ----- Model selection -----
class NN_concat(nn.Module):
    def __init__(self, params, input_size_1, input_size_2, input_size_3):
        super(NN_concat, self).__init__()

        if args.dataset == 'ROSMAP':
            dim1, dim2, dim3 = 200, 200, 100
        else:
            dim1, dim2, dim3 = 400, 400, 200

        self.block1 = nn.Sequential(
            nn.Linear(input_size_1, dim1), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim1),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim2, dim3), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim3),
        )

        self.block2 = nn.Sequential(
            nn.Linear(input_size_2, dim1),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim1),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim1, dim2), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim2),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim2, dim3), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim3),
        )

        self.block3 = nn.Sequential(
            nn.Linear(input_size_3, dim1),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim1),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim1, dim2), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim2),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim2, dim3),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim3),
        )

        if args.dataset == 'ROSMAP':
            self.out = nn.Sequential(nn.Linear(dim3*3, 100),
                                     nn.LeakyReLU(0.25),
                                     nn.BatchNorm1d(100),
                                     nn.Dropout(params['dropout1']), 
                                     nn.Linear(100, 2))
        else:
            self.out = nn.Sequential(nn.Linear(dim3*3, 100),
                                     nn.LeakyReLU(0.25),
                                     nn.BatchNorm1d(100),
                                     nn.Dropout(params['dropout1']), 
                                     nn.Linear(100, 5))

    def forward(self, input_1, input_2, input_3):
        # 3 block, 1 for each omic
        out1 = self.block1(input_1)
        out2 = self.block2(input_2)
        out3 = self.block3(input_3)
        

        # combining the 3 representations, concatenating the vectors
        combined = torch.cat((out1.view(out1.size(0), -1),
                             out2.view(out2.size(0), -1),
                             out3.view(out3.size(0), -1)), dim=1)
        
        combined_out = self.out(combined)

        return combined_out
    

class CrossModal_NN_concat(nn.Module):
    def __init__(self, params, input_size_1, input_size_2, input_size_3):
        super(CrossModal_NN_concat, self).__init__()

        if args.dataset == 'ROSMAP':
            dim1, dim2, dim3, dim4 = 200, 200, 100, 100
        else:
            dim1, dim2, dim3, dim4 = 400, 400, 200, 200

        self.first_block1 = nn.Sequential(
            nn.Linear(input_size_1, dim1), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim1),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim1, dim2),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim2),

        )
        self.connection1 = nn.Sequential(nn.Linear(dim2, dim3)) 

        self.second_block1 = nn.Sequential(
            nn.Dropout(params['dropout1']),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim3*3),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim3*3, dim4),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim4)
        )
        
        self.first_block2 = nn.Sequential(
            nn.Linear(input_size_2, dim1),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim1),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim1, dim2), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim2),
        )
        self.connection2 = nn.Sequential(nn.Linear(dim2, dim3))

        self.second_block2 = nn.Sequential(
            nn.Dropout(params['dropout1']),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim3*3),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim3*3, dim4),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim4)
        )
        
        self.first_block3 = nn.Sequential(
            nn.Linear(input_size_3, dim1), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim1),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim1, dim2), 
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim2),
        )
        self.connection3 = nn.Sequential(nn.Linear(dim2, dim3))

        self.second_block3 = nn.Sequential(
            nn.Dropout(params['dropout1']),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim3*3),
            nn.Dropout(params['dropout1']),
            nn.Linear(dim3*3, dim4),
            nn.LeakyReLU(0.25),
            nn.BatchNorm1d(dim4)
        )

        
        self.cross_connection12 = nn.Sequential(nn.Linear(dim2, dim3))
        self.cross_connection32 = nn.Sequential(nn.Linear(dim2, dim3))
        self.cross_connection21 = nn.Sequential(nn.Linear(dim2, dim3))
        self.cross_connection23 = nn.Sequential(nn.Linear(dim2, dim3))
        self.cross_connection13 = nn.Sequential(nn.Linear(dim2, dim3))
        self.cross_connection31 = nn.Sequential(nn.Linear(dim2, dim3))

        if args.dataset == 'ROSMAP':
            self.out = nn.Sequential(nn.Linear(dim4*3, 100),
                                     nn.LeakyReLU(0.25),
                                     nn.BatchNorm1d(100),
                                     nn.Dropout(params['dropout1']), 
                                     nn.Linear(100, 2))
        else:
            self.out = nn.Sequential(nn.Linear(dim4*3, 100),
                                     nn.LeakyReLU(0.25),
                                     nn.BatchNorm1d(100),
                                     nn.Dropout(params['dropout1']), 
                                     nn.Linear(100, 5))

    def forward(self, input_1, input_2, input_3):
        # 3 block, 1 for each omic
        out1 = self.first_block1(input_1)
        out2 = self.first_block2(input_2)
        out3 = self.first_block3(input_3)

        # cross connection for each view(branch)
        out_connection1 = self.connection1(out1) 
        out_cross_connection21 = self.cross_connection21(out2)
        out_cross_connection31 = self.cross_connection31(out3)
        
        out_connection2 = self.connection2(out2) 
        out_cross_connection12 = self.cross_connection12(out1)
        out_cross_connection32 = self.cross_connection32(out3)
        
        out_connection3 = self.connection3(out3) 
        out_cross_connection13 = self.cross_connection13(out1)
        out_cross_connection23 = self.cross_connection23(out2)

        # cross conncections: between first and second block

        crossed1 = torch.cat((out_connection1.view(out_connection1.size(0), -1),
                            out_cross_connection21.view(out_cross_connection21.size(0), -1),
                            out_cross_connection31.view(out_cross_connection31.size(0), -1)), dim=1)
        out11 = self.second_block1(crossed1)


        crossed2 = torch.cat((out_cross_connection12.view(out_cross_connection12.size(0), -1),
                            out_connection2.view(out_connection2.size(0), -1),
                            out_cross_connection32.view(out_cross_connection32.size(0), -1)), dim=1)
        out22 = self.second_block2(crossed2)


        crossed3 = torch.cat((out_cross_connection13.view(out_cross_connection13.size(0), -1),
                            out_cross_connection23.view(out_cross_connection23.size(0), -1),
                            out_connection3.view(out_connection3.size(0), -1)), dim=1)
        out33 = self.second_block3(crossed3)
        

        # combining the 3 representations, concatenating the vectors
        combined = torch.cat((out11.view(out11.size(0), -1),
                            out22.view(out22.size(0), -1),
                            out33.view(out33.size(0), -1)), dim=1)
        
        combined_out = self.out(combined)

        return combined_out



def load_model(params, input_size_1, input_size_2, input_size_3, model_name=args.model):
    # Model selected
    if model_name == 'NN_concat':
        model = NN_concat(params, input_size_1, input_size_2, input_size_3)
    else:
        model = CrossModal_NN_concat(params, input_size_1, input_size_2, input_size_3)

    model.double() 
    model = model.to(args.device)
    return model

if __name__ == "__main__":
    pass


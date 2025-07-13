import torch.nn as nn

def create_mlp(input_dim, hidden_dims, output_dim = None):

        layers = []
        in_features = input_dim
        layers_dim = hidden_dims if output_dim is None else hidden_dims + [output_dim]

        for out_features in layers_dim:

            linear = nn.Linear(in_features, out_features)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu') 
            nn.init.zeros_(linear.bias)
            
            layers.extend([
                linear,
                nn.BatchNorm1d(out_features),
                nn.ReLU()
            ])
            in_features = out_features

        return nn.Sequential(*layers[: -2]) #remove the last 2 elements (BatchNorm and ReLU)
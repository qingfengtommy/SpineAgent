import torch

def check_nan_weights(state_dict, ckpt_name):
        print(f"Checking {ckpt_name} ...")
        nan_layers = []
        min_value = float('inf')
        max_value = float('-inf')
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"NaN found in layer: {name}")
                nan_layers.append(name)
            min_value = min(min_value, param.min())
            max_value = max(max_value, param.max())
        print(f"Min value: {min_value}, Max value: {max_value}")
        if not nan_layers:
            print(f"No NaNs found in {ckpt_name}.")
        return nan_layers

path_t1 = "/home/hzhanguw/research-projects/CLIPlogs/t1_full_epoch20_img60_lr1e-4_context512_Dino_BiomedBERT/checkpoints/epoch_latest.pt"
path_t2 = "/home/hzhanguw/research-projects/CLIPlogs/t2_full_epoch20_img60_lr1e-4_context512_Dino_BiomedBERT/checkpoints/epoch_latest.pt"
state_dict_t1 = torch.load(path_t1)['state_dict']
state_dict_t2 = torch.load(path_t2)['state_dict']

nan_layers_t1 = check_nan_weights(state_dict_t1, "T1")
nan_layers_t2 = check_nan_weights(state_dict_t2, "T2")
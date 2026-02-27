import os
import torch
from torch.utils.data import DataLoader
import h5py
import numpy as np


def l2_loss(pred, true):
    loss = torch.sum((pred-true)**2, dim=[1, 2, 3])
    return torch.mean(loss)


def get_train_data(args):
    
    frac_data = h5py.File('Binary_training.mat','r')
    fracture = frac_data['Binary']
    fracture = np.array(fracture).transpose((2,1,0))
    fracture = fracture[0:args.train_number,np.newaxis,:,:]
    fracture = np.repeat(fracture, 11, axis=1)
    frac_data.close()

    pred_data = h5py.File('state_training.mat','r')
    state = pred_data['state_training']
    state = np.array(state).transpose((3,0,2,1))
    state = state[0:args.train_number,:,:,:]
    pred_data.close()
    
    t = np.linspace(-1, 1, 11)
    # t = np.repeat(np.arange(1, 12), 2)
    t = t[np.newaxis, :, np.newaxis, np.newaxis]
    t_step = np.repeat(t, args.train_number, axis=0)
    t_step = np.repeat(t_step, 128, axis=2)
    t_step = np.repeat(t_step, 128, axis=3)
    
    frac_norm = fracture.reshape(-1, 1, 128, 128)
    t_norm = t_step.reshape(-1, 1, 128, 128)
    state_norm = state.reshape(-1, 1, 128, 128)
    x = np.concatenate((frac_norm, t_norm), 1)
    y = state_norm
    
    indices = np.random.permutation(len(x))
    x, y = x[indices], y[indices]
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(y, x)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    # field, condition = load_data(args.train_number)
    # field = torch.as_tensor(field)
    # condition = torch.as_tensor(condition)
    # dataset = torch.utils.data.TensorDataset(field, condition)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)

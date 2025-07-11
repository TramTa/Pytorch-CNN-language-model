import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot
import numpy as np 



max_len=250

def train(args):
    from os import path
    import torch.utils.tensorboard as tb

    model = TCN()

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th')))

    loss = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_data = torch.utils.data.DataLoader(SpeechDataset('data/train.txt', transform=one_hot), batch_size=args.batch_size, shuffle=True)
    valid_data = torch.utils.data.DataLoader(SpeechDataset('data/valid.txt', transform=one_hot), batch_size=args.batch_size, shuffle=True)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        losses = []

        for item in train_data:
            sen = item[:,:,:-1].to(device)
            label = item.argmax(dim=1).to(device)   

            pred = model(sen)
            loss_val = loss(pred, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            
            losses.append(loss_val.detach().cpu().numpy())
        
        avg_loss = np.mean(losses)
        print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
        save_model(model)

    save_model(model)








if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    args = parser.parse_args()
    train(args)

import json
import sys
import os
import argparse
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.Text_Recognization.text_recognization import *
from src.Text_Recognization.prepare_dataset import *
from src.Text_Recognization.dataloader import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_json_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config

def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for images, labels, labels_len in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            logits_lens = torch.full(
                size=(outputs.size(1), ),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)
            
            loss = criterion(outputs, labels, logits_lens, labels_len)
            losses.append(loss.item())
            
        eval_loss = sum(losses) / len(losses)
        return eval_loss
        

def training_loop(model, train_loader, val_loader, learning_rate, epochs, optimizer, criterion, scheduler, device):
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        
        batch_losses = []
        for images, labels, labels_len in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            logits_lens = torch.full(
                size=(outputs.size(1), ),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)
            
            loss = criterion(outputs, labels, logits_lens, labels_len)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(train_loss)
        
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"epoch: {epoch+1}/{epochs}\ttrain_loss:{train_loss}\tval_loss:{val_loss}")
        
        scheduler.step()
        
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=os.getcwd(), help='Path to the root directory')
    
    args = parser.parse_args()
    config_path = 'src/config.json'
    root_path = os.path.join(args.root_path, 'Dataset')
    config = load_json_config(config_path)

    # dictionary char and idx
    char_to_idx, idx_to_char = build_vocab(root_path)

    # model
    model = CRNN(vocab_size=config['vocab_size'], hidden_size=config['CRNN']['hidden_size'], n_layers=config['CRNN']['n_layers'])

    # dataloader
    train_loader, val_loader, test_loader = get_dataloader()

    # define hyper parammeters
    criterion = nn.CTCLoss(
        blank=char_to_idx[config['blank_char']],
        zero_infinity=True,
        reduction='mean'
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['CRNN']['learning_rate'],
        weight_decay=config['CRNN']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config['CRNN']['scheduler_step_size'],
        gamma=0.1
    )
    
    # training loop
    train_losses, val_losses = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['CRNN']['learning_rate'],
        epochs=config['CRNN']['epochs'],
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device
    )
    
if __name__ == '__main__':
    main()
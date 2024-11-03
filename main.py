import torch
import wandb
import numpy as np

import time
import os

from Utils.data_utils import read_config
from Utils.training_utils import Optimizer, Criterion, Scheduler
from model.attention import VisionTransformer
from model.ResNet import *
from Dataset.dataset import train_test_split_loader


def main():
    config = read_config('config/config.yaml')
    wandb_config = read_config('config/wandb_config.yaml')
    
    print("Initializing!")

    # W & B login and initialization
    need_wandb = config['Visualization']['wandb']
    if need_wandb:
        wandb.login(key=wandb_config['API_key'])
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            name=wandb_config['name'],
            tags=wandb_config['tags'],
            notes=wandb_config['notes'],
            dir=wandb_config['dir'],
            config=config
        )

    # Matplotlib initialization
    need_matplotlib = config['Visualization']['matplotlib']
    if need_matplotlib:
        train_loss, train_acc = [], []
        test_loss, test_acc = [], []

    print('============================== Dataset Details =====================================')
    # Create Dataset
    dataset_name = config['Dataset']['dataset_name']
    print(f"Dataset Name: {dataset_name}")
    print(f"Train Ratio: {config['Dataset'][dataset_name]['train_ratio']}, Image Size: {config['Dataset'][dataset_name]['image_size']}")
    print(f"Batch Size: {config['Dataset'][dataset_name]['batch_size']}, Classes Number: {config['Dataset'][dataset_name]['num_classes']}")
    train_loader, test_loader = train_test_split_loader(config, dataset_name)
    time.sleep(2) # pause 2 sec to check

    # Create Models
    print('============================== Model Details ======================================')
    model_name = config['Model']['model_name']
    retrain = config['Model']['retrain']
    if not os.path.exists(config['Model']['save_directory']):
        os.makedirs(config['Model']['save_directory'])
    model_save_path = os.path.join(config['Model']['save_directory'], config['Model'][model_name]['save_model_name'])
    print(f"Model Name: {model_name}")
    print(f"Weight Init: {config['Model'][model_name]['weight_init']}, Dropout Rate: {config['Model'][model_name]['dropout_rate']}")

    if model_name == 'ViT':
        print(f"Num Heads: {config['Model']['ViT']['num_heads']}, Embed Dim: {config['Model']['ViT']['embed_dim']}")
        print(f"Patch Size: {config['Model']['ViT']['patch_size']}, Num Blocks: {config['Model']['ViT']['num_blocks']}")
        model = VisionTransformer(config['Dataset'][dataset_name]['in_channels'],
                                config['Model'][model_name]['patch_size'], 
                                config['Dataset'][dataset_name]['image_size'],
                                config['Model'][model_name]['embed_dim'], 
                                config['Model'][model_name]['num_heads'], 
                                config['Model'][model_name]['num_blocks'], 
                                config['Dataset'][dataset_name]['num_classes'],
                                config['Model'][model_name]['dropout_rate']
                                )
    elif model_name == 'ResNet':
        print(f"ResNet Type: {config['Model']['ResNet']['ResNet_type']}")
        model = get_ResNet(config)

    print(f"Retrain: {retrain}")
    if retrain:
        model.load_state_dict(torch.load(model_save_path))
    time.sleep(2) 

    # Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = Criterion(config)
    optimizer = Optimizer(config, model.parameters())
    scheduler = Scheduler(config, optimizer)

    print("Training Start!")
    print(f"Total num_epoch: {config['Training']['num_epoch']}")
    num_epoch = config['Training']['num_epoch']

    for epoch in range(num_epoch):
        print(f"epoch: {epoch + 1}")
        print("Training Section")
        model.train()
        loss_list = []
        acc_list = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = torch.argmax(output, dim=-1)
                acc = (pred == labels).sum() / labels.size(0)
                acc_list.append(acc.item())
                loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()
        train_loss_avg = np.mean(loss_list)
        train_acc_avg  = np.mean(acc_list)
        print(f'train loss = {train_loss_avg}, train accuracy = {train_acc_avg}')

        print("Test Section")
        model.eval()
        test_loss_list = []
        test_acc_list = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = criterion(out, labels)
                pred = torch.argmax(out, dim=-1)
                acc = (pred == labels).sum().item() / labels.size(0)
                test_acc_list.append(acc)
                test_loss_list.append(loss.item())

        test_loss_avg = np.mean(test_loss_list)
        test_acc_avg = np.mean(test_acc_list)
        print(f'test loss = {test_loss_avg}, test accuracy = {test_acc_avg}')

        if need_wandb:
            wandb.log({"train_loss": train_loss_avg, "train_accuracy": train_acc_avg})
            wandb.log({"test_loss": test_loss_avg, "test_accuracy": test_acc_avg})

        if need_matplotlib:
            train_loss.append(train_loss_avg)
            train_acc.append(train_acc_avg)
            test_loss.append(test_loss_avg)
            test_acc.append(test_acc_avg)


    # Finish Training
    print("Training Finished!")
    torch.save(model.state_dict(), model_save_path)

    if need_wandb:
        wandb.save(model_save_path)
        wandb.finish()

    if need_matplotlib:
        import matplotlib.pyplot as plt
        if not os.path.exists(config['Visualization']['save_fig_directory']):
            os.makedirs(config['Visualization']['save_fig_directory'])
        plt.plot(train_loss, color='blue', label='train loss')
        plt.plot(test_loss, color='red', label='test loss')
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(config['Visualization']['save_fig_directory'], 'loss_diagram.png'))
        plt.clf()

        plt.plot(train_acc, color='blue', label='train acc')
        plt.plot(test_acc, color='red', label='test acc')
        plt.xlabel('steps')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(config['Visualization']['save_fig_directory'], 'acc_diagram.png'))



if __name__ == '__main__':
    main()
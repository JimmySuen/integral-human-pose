import torch.optim

def get_optimizer(config, network):
    if config.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)
    elif config.optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.wd) 
    else:
        print("Error! Unknown optimizer name: ", config.optimizer_name)
        assert 0

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(i) for i in config.lr_epoch_step.split(',')],
                                                     gamma=config.lr_factor)
    return optimizer, scheduler
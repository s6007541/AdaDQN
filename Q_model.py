import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, env, args=None):
        super().__init__()
        if args == None:
            depth = 0
        else:
            depth = args.network_depth
        
        base_layers = [nn.Conv2d(4, 32, 8, stride=4),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 4, stride=2),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),]
        
        stack_layers = []
        num_node = 64
        for d in range(depth+1):
           
            num_next_node = num_node if d == depth else num_node*2
            stack_layers += [nn.Conv2d(num_node, num_next_node, 3, stride=1),
                            nn.BatchNorm2d(num_next_node),
                            nn.ReLU(),]
            num_node *= 2
        
        classification_layers = [nn.Flatten(),
                                nn.Linear(num_node//2 * ((7 - depth*2)**2), 512),
                                nn.ReLU(),
                                nn.Linear(512, env.single_action_space.n)]

        all_layers = base_layers + stack_layers + classification_layers
        self.network = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.network(x / 255.0)
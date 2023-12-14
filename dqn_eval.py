import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import argparse
from corruptions import *
import pathlib
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

def covirate_shift(x, args):
    if args.corruption_type in CORRUPTION_LIST:
        corruption_func = CORRUPTION_LIST[args.corruption_type]
    else:
        raise NotImplementedError

    return corruption_func(x, args.corruption_level)
        
def AdaDQN_init(model):
    # turn on grad for BN params only

    for param in model.parameters():  # initially turn off requires_grad for all
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
            # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
           
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    memory = []
    return model, optimizer, memory


def evaluate(
    model_path,
    make_env,
    env_id,
    eval_episode,
    run_name,
    Model,
    device = torch.device("cpu"),
    epsilon = 0.05,
    capture_video= True,
    args = None
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs,args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    if args.AdaDQN:
        model, optimizer, memory = AdaDQN_init(model)
        
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episode:
        
        # corruptions
        obs = covirate_shift(obs, args)
        model.eval()
        obs_tensor = torch.Tensor(obs).to(device)
        q_values = model(obs_tensor)
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
                
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
        
    from dqn_atari import make_env
    from Q_model import QNetwork
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint_path', type=str, default='runs/ALE/AirRaid-v5__AirRaid-v5_5M_BN__1__1699933828/AirRaid-v5_5M_BN.pth', help='path to .pth file')
    parser.add_argument('--game_name', type=str, default='ALE/AirRaid-v5', help='name of atari game')   
    parser.add_argument('--AdaDQN', action='store_true', default=True, help="activate AdaDQN algorithm")   
    parser.add_argument('--eval_eps',default=10, type=int, help='number of eps')
    parser.add_argument('--corruption_type', type=str, default='none', help='type of corruption')
    parser.add_argument('--corruption_level',default=1, type=int, help='level of severity of corruption')
    parser.add_argument('--device_id', default = 1, type = int, help='gpu device id')
    parser.add_argument('--network_depth', default = 0, type = int, help='depth of QNetwork, must be similar to training process')
    parser.add_argument("--debug", action='store_true', default=False, help="debug mode")
    
    args = parser.parse_args()
    if args.debug:
        import debugpy
        debugpy.listen(5679)
        print("wait for debugger")
        debugpy.wait_for_client()
        print("attach")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device :', device)
    torch.cuda.set_device(args.device_id)

    model_path = args.checkpoint_path
    final_returns = evaluate(model_path, make_env, args.game_name, eval_episode=args.eval_eps, run_name=f"eval", Model=QNetwork, device=device, capture_video=False, args=args)    
    print("average returns :", sum(final_returns)/len(final_returns))
    
    pathlib.Path(f'results/{args.game_name}').mkdir(parents=True, exist_ok=True) 

    if args.AdaDQN:
        file_name = f'results/{args.game_name}/AdaDQN_{args.corruption_type}_{args.corruption_level}.txt'
    else:
        file_name = f'results/{args.game_name}/DQN_{args.corruption_type}_{args.corruption_level}.txt'
   
    with open(file_name, 'w') as f:
        ls = [float(l[0]) for l in final_returns]
        f.write(str(ls) + "\n")
        f.write(str((sum(ls)/len(ls))))
    
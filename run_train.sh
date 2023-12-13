python dqn_atari.py \
    --exp-name MsPacman-v5_3M_BN_depth2 \
    --capture-video \
    --env-id ALE/MsPacman-v5 \
    --total-timesteps 3000000 \
    --buffer-size 400000 \
    --save-model \
    --network_depth 2\

python dqn_atari.py \
    --exp-name AirRaid-v5_3M_BN_depth2 \
    --capture-video \
    --env-id ALE/AirRaid-v5 \
    --total-timesteps 3000000 \
    --buffer-size 400000 \
    --save-model \
    --network_depth 2\

python dqn_atari.py \
    --exp-name Phoenix-v5_3M_BN_depth2 \
    --capture-video \
    --env-id ALE/Phoenix-v5 \
    --total-timesteps 3000000 \
    --buffer-size 400000 \
    --save-model \
    --network_depth 2\

python dqn_atari.py \
    --exp-name SpaceInvaders-v5_3M_BN_depth2 \
    --capture-video \
    --env-id ALE/SpaceInvaders-v5 \
    --total-timesteps 3000000 \
    --buffer-size 400000 \
    --save-model \
    --network_depth 2\



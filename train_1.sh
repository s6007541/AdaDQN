# python dqn_atari.py \
#     --exp-name Breakout-v5_10M_BN \
#     --capture-video \
#     --env-id ALE/Breakout-v5 \
#     --total-timesteps 10000000 \
#     --buffer-size 400000 \
#     --save-model \

python dqn_atari.py \
    --exp-name MsPacman-v5_2M_BN_depth2 \
    --capture-video \
    --env-id ALE/MsPacman-v5 \
    --total-timesteps 2000000 \
    --buffer-size 400000 \
    --save-model \
    --network_depth 2\

    # < 1000
# python dqn_atari.py \
#     --exp-name AirRaid-v5_5M_BN \
#     --capture-video \
#     --env-id ALE/AirRaid-v5 \
#     --total-timesteps 2000000 \
#     --buffer-size 400000 \
#     --save-model \
#     --network_depth 2 &
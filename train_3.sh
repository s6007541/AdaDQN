python dqn_atari.py \
    --exp-name AirRaid-v5_2M_BN_d0 \
    --capture-video \
    --env-id ALE/AirRaid-v5 \
    --total-timesteps 2000000 \
    --buffer-size 400000 \
    --save-model \
    --network_depth 0\

    # < 100
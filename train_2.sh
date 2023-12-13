python dqn_atari.py \
    --exp-name Centipede-v5_2M_BN_depth2 \
    --capture-video \
    --env-id ALE/Centipede-v5 \
    --total-timesteps 2000000 \
    --buffer-size 400000 \
    --save-model \
    --network_depth 2\

    # < 1000
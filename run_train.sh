GAMES=("MsPacman" "AirRaid" "Phoenix" "SpaceInvaders")

for GAME in "${GAMES[@]}"; do
    python -W ignore dqn_atari.py \
            --exp-name ${GAMES}-v5_3M_BN_depth2 \
            --env-id ALE/${GAMES}-v5 \
            --total-timesteps 10000 \
            --buffer-size 400000 \
            --save-model \
            --network_depth 2\
done


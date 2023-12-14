CORRUPTIONS=("none" "gaussian_noise" "shot_noise" "impulse_noise" "speckle_noise" "gaussian_blur" "defocus_blur" "contrast" "brightness" "saturate")
CORRUPTIONS_LEVEL=(1 2 3 4 5)


GAME="" # fill this : ie ALE/Phoenix-v5
CKPTPATH="" # fill this : ie runs/ALE/Phoenix-v5__Phoenix-v5_2M_BN_depth1__1__1700541104/Phoenix-v5_2M_BN_depth1.pth

for C in "${CORRUPTIONS[@]}"; do
    for C_LVL in "${CORRUPTIONS_LEVEL[@]}"; do
        python -W ignore dqn_eval.py --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 1000 --corruption_level ${C_LVL} --device_id 0 --network_depth 2
        python -W ignore dqn_eval.py --AdaDQN --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 1000 --corruption_level ${C_LVL} --device_id 0 --network_depth 2

    done
done
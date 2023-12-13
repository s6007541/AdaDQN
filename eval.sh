# python dqn_eval.py --tta TENT --corruption_type none
# python dqn_eval.py --tta TENT --corruption_type gaussian_noise --corruption_level 5
# python dqn_eval.py --tta PL --corruption_type gaussian_noise --corruption_level 5
CORRUPTIONS=("gaussian_blur" )
# CORRUPTIONS=("none" "gaussian_noise" "shot_noise" "impulse_noise" "speckle_noise" "gaussian_blur" "defocus_blur" "contrast" "brightness" "saturate")
# GAME="ALE/SpaceInvaders-v5"
# CKPTPATH="/mnt/sting/sorn111930/atari/atari-dqn/runs/ALE/SpaceInvaders-v5__SpaceInvaders-v5_2M_BN__1__1700103603/SpaceInvaders-v5_2M_BN.pth"
# GAME="ALE/AirRaid-v5"
# CKPTPATH="/mnt/sting/sorn111930/atari/atari-dqn/runs/ALE/AirRaid-v5__AirRaid-v5_5M_BN__1__1699933828/AirRaid-v5_5M_BN.pth"
# GAME="ALE/Phoenix-v5"
# CKPTPATH="/mnt/sting/sorn111930/atari/atari-dqn/runs/ALE/Phoenix-v5__Phoenix-v5_2M_BN_depth2__1__1700415385/Phoenix-v5_2M_BN_depth2.pth"
# GAME="ALE/Centipede-v5"
# CKPTPATH="/mnt/sting/sorn111930/atari/atari-dqn/runs/ALE/Centipede-v5__Centipede-v5_2M_BN_depth2__1__1700411218/Centipede-v5_2M_BN_depth2.pth"
# GAME="ALE/VideoPinball-v5"
# CKPTPATH="/mnt/sting/sorn111930/atari/atari-dqn/runs/ALE/VideoPinball-v5__VideoPinball-v5_2M_BN_depth2__1__1700415354/VideoPinball-v5_2M_BN_depth2.pth"

# GAME="ALE/Phoenix-v5"
# CKPTPATH="/mnt/sting/sorn111930/atari/atari-dqn/runs/ALE/Phoenix-v5__Phoenix-v5_2M_BN_depth0__1__1700541099/Phoenix-v5_2M_BN_depth0.pth"
# GAME="ALE/Phoenix-v5"
# CKPTPATH="/mnt/sting/sorn111930/atari/atari-dqn/runs/ALE/Phoenix-v5__Phoenix-v5_2M_BN_depth1__1__1700541104/Phoenix-v5_2M_BN_depth1.pth"
GAME="ALE/Phoenix-v5"
CKPTPATH="/mnt/sting/sorn111930/atari/atari-dqn/runs/ALE/Phoenix-v5__Phoenix-v5_2M_BN_depth2__1__1700415385/Phoenix-v5_2M_BN_depth2.pth"

for C in "${CORRUPTIONS[@]}"; do
    python -W ignore dqn_eval.py --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 1 --device_id 5 --network_depth 2&
    python -W ignore dqn_eval.py --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 2 --device_id 5 --network_depth 2&
    python -W ignore dqn_eval.py --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 3 --device_id 5 --network_depth 2&
    python -W ignore dqn_eval.py --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 4 --device_id 5 --network_depth 2&
    python -W ignore dqn_eval.py --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 5 --device_id 5 --network_depth 2&

    python -W ignore dqn_eval.py --tta bn_stats --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 1 --device_id 6 --network_depth 2&
    python -W ignore dqn_eval.py --tta bn_stats --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 2 --device_id 6 --network_depth 2&
    python -W ignore dqn_eval.py --tta bn_stats --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 3 --device_id 6 --network_depth 2&
    python -W ignore dqn_eval.py --tta bn_stats --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 4 --device_id 6 --network_depth 2&
    python -W ignore dqn_eval.py --tta bn_stats --checkpoint_path ${CKPTPATH} --corruption_type ${C} --game_name ${GAME} --eval_eps 100 --corruption_level 5 --device_id 6 --network_depth 2
    
done
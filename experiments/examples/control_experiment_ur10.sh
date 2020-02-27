enable_oods=( 'True' 'False' )
ood_name="drop_2_clean_2"

python3 ../control_experiment_ur10.py \
    --camera_res=3,480,640 \
    --dbname="/home/olimoyo/robust-latent-srl/experiments/results/test_tcp.db" \
    --goal_path="/home/olimoyo/robust-latent-srl/experiments/64x64_white_2_goal.npy" \
    --dt=0.5 \
    --speed_max=0.5 \
    --timeout=30 \
    --mpc_horizon=3 \
    --enable_ood=False \
    --hosts=192.168.42.115 \
    --ports=5000 \
    --num_episodes=9 \
    --args_output_file="/home/olimoyo/robust-latent-srl/experiments/results/test_tcp_params.pkl" \
    --repeat_actions=3 \
    --seed=0 \
    --render=False \
    --model_path="/home/olimoyo/robust-latent-srl/srl/saved_models/real-world_reacher/iros/64x64/02-16-20_17:03:55_res64_netgru_constant_a4_z10_trajlen7_wd0_bs64_bnFalse_lr3e-4_R0.03" \
    --device=cuda:0 \
    --T=4
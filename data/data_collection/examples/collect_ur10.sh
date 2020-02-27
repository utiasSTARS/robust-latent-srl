# python3 /home/olimoyo/multimodal-latent-srl/real_sense_server/tcp_server.py --host=localhost --port=5000 --device_id=0 --height=480 --width=640 --frame_rate=30 --colour_format=rgb8 &
python3 collect_ur10.py \
    --camera_res=3,480,640 \
    --dbname="/home/olimoyo/robust-latent-srl/experiments/results/1024T_white_3.db" \
    --dt=0.5 \
    --speed_max=0.5 \
    --timeout=15 \
    --hosts=192.168.42.115 \
    --ports=5000 \
    --num_episodes=1024 \
    --args_output_file="/home/olimoyo/robust-latent-srl/experiments/results/1024T_white_3_params.pkl" \
    --repeat_actions=3 \
    --seed=0 \
    --render=False

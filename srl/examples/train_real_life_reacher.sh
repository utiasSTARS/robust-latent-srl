device="cuda:0"
dataset="path/to/cache/<name>.pkl"

n_batches=(64)
learning_rates=(3e-4)
schedulers=('none')
batch_norms=('False')
bi_directionals=('False')
weight_inits=('custom')
ks=(15)
alpha_nets=('gru')
n_epochs=(4096)
opt=('adam')
folder=('traj_16_no_BN_post_noise_p_1e-2')
storage_base_path="path/to/result/store/${folder}/"
task="real_life_reacher"
measurement_net=('cnn')
debug=('False')
nl=('relu')
free_nats=(0)
traj_lens=(7)
init_cov=(20.0)
measurement_uncertainties=('learn_separate' 'scale')
weight_decays=(0 1e-5)
val_split=0.07
dim_zs=(10)
dim_as=(4)
emission_noises=(0.03)

for n in {1..1}; do
    lam_rec=0.95
    lam_kl=0.80
    n_annealing_epoch_beta=0
    opt_vae_kf_epoch=1024
    for scheduler in ${schedulers[@]}; do
        for batch_norm in ${batch_norms[@]}; do
            for k in ${ks[@]}; do
                for weight_init in ${weight_inits[@]}; do
                    for weight_decay in ${weight_decays[@]}; do
                    for n_batch in ${n_batches[@]}; do
                        for alpha_net in ${alpha_nets[@]}; do
                            for lr in ${learning_rates[@]}; do
                                for bi_directional in ${bi_directionals[@]}; do
                                    for n_epoch in ${n_epochs[@]}; do
                                        for emission_noise in ${emission_noises[@]}; do
                                        for measurement_uncertainty in ${measurement_uncertainties[@]}; do
                                        for traj_len in ${traj_lens[@]}; do
                                        for dim_a in ${dim_as[@]}; do
                                        for dim_z in ${dim_zs[@]}; do
                                            python3 ../srl/train.py \
                                                                --k $k \
                                                                --dim_a $dim_a \
                                                                --dim_z $dim_z \
                                                                --dim_u 2 \
                                                                --dim_alpha $dim_z \
                                                                --n_worker 0 \
                                                                --use_binary_ce "False" \
                                                                --beta1 0.9 \
                                                                --beta2 0.999 \
                                                                --n_epoch $n_epoch \
                                                                --debug $debug \
                                                                --n_batch $n_batch \
                                                                --device $device \
                                                                --lr $lr \
                                                                --weight_init $weight_init \
                                                                --weight_decay $weight_decay \
                                                                --dataset $dataset \
                                                                --lam_rec $lam_rec \
                                                                --lam_kl $lam_kl \
                                                                --comment "res64_net${alpha_net}_${measurement_uncertainty}_a${dim_a}_z${dim_z}_trajlen${traj_len}_wd${weight_decay}_bs${n_batch}_bn${batch_norm}_lr${lr}_R${emission_noise}" \
                                                                --storage_base_path "${storage_base_path}" \
                                                                --scheduler $scheduler \
                                                                --fc_hidden_size 128 \
                                                                --alpha_hidden_size 128 \
                                                                --use_bidirectional $bi_directional \
                                                                --use_batch_norm $batch_norm \
                                                                --measurement_net $measurement_net \
                                                                --transition_noise 0.08 \
                                                                --emission_noise $emission_noise \
                                                                --opt_vae_epochs 0 \
                                                                --opt_vae_kf_epochs $opt_vae_kf_epoch \
                                                                --n_annealing_epoch_beta $n_annealing_epoch_beta \
                                                                --opt $opt \
                                                                --alpha_net $alpha_net \
                                                                --task $task \
                                                                --val_split $val_split \
                                                                --dim_x "1,64,64" \
                                                                --non_linearity $nl \
                                                                --traj_len $traj_len \
                                                                --init_cov $init_cov \
                                                                --free_nats $free_nats \
                                                                --measurement_uncertainty $measurement_uncertainty
                                            done
                                            done
					    # wait
                                            done
                                            done
                                    done
                                    done
                                done
                            done
                            done
                        done
                    done
                done
            done
        done
    done
done

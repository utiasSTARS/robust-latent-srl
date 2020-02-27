device="cuda:0"
dataset="path/to/cache/<name>.pkl"

n_batches=(64)
decay_learning_rates=(3e-4)
schedulers=('none')
batch_norms=('False')
bi_directionals=('False')
weight_inits=('custom')
ks=(15)
alpha_nets=('lstm')
n_epochs=(16384)
opt=('adam')
comment=('traj32_learnseparate_NoRegularize_expstacked_Q0.08')
storage_base_path="path/to/result/store/${comment}/"
resume_training_path=("none")
measurement_net=('cnn')
debug=('True')
nl=('relu')
free_nats=(0)
traj_len=(32)
init_cov=(20.0)
measurement_uncertainties=('learn_separate')
weight_decay=(1e-5)
use_stochastic_dynamics=('False')

for n in {1..1}; do
    lam_rec=0.95
    lam_kl=0.80
    n_annealing_epoch_beta=0
    opt_vae_epoch=0
    opt_vae_kf_epoch=2048
    for scheduler in ${schedulers[@]}; do
        for batch_norm in ${batch_norms[@]}; do
            for k in ${ks[@]}; do
                for weight_init in ${weight_inits[@]}; do
                    for n_batch in ${n_batches[@]}; do
                        for alpha_net in ${alpha_nets[@]}; do
                            for lr in ${decay_learning_rates[@]}; do
                                for bi_directional in ${bi_directionals[@]}; do
                                    for n_epoch in ${n_epochs[@]}; do
                                        for measurement_uncertainty in ${measurement_uncertainties[@]}; do
                                            python3 ../srl/train.py \
                                                                --k $k \
                                                                --dim_a 2 \
                                                                --dim_z 3 \
                                                                --dim_alpha 3 \
                                                                --n_worker 4 \
                                                                --use_binary_ce "False" \
                                                                --beta1 0.9 \
                                                                --beta2 0.999 \
                                                                --n_epoch $n_epoch \
                                                                --debug $debug \
                                                                --comment $comment \
                                                                --n_batch $n_batch \
                                                                --device $device \
                                                                --lr $lr \
                                                                --weight_init $weight_init \
                                                                --dataset $dataset \
                                                                --lam_rec $lam_rec \
                                                                --lam_kl $lam_kl \
                                                                --storage_base_path $storage_base_path \
                                                                --resume_training_path $resume_training_path \
                                                                --scheduler $scheduler \
                                                                --fc_hidden_size 128 \
                                                                --alpha_hidden_size 128 \
                                                                --use_bidirectional $bi_directional \
                                                                --use_batch_norm $batch_norm \
                                                                --measurement_net $measurement_net \
                                                                --transition_noise 0.08 \
                                                                --emission_noise 1.0 \
                                                                --opt_vae_epochs $opt_vae_epoch \
                                                                --opt_vae_kf_epochs $opt_vae_kf_epoch \
                                                                --n_annealing_epoch_beta $n_annealing_epoch_beta \
                                                                --opt $opt \
                                                                --alpha_net $alpha_net \
                                                                --task "pendulum64" \
                                                                --val_split 0 \
                                                                --dim_x "1,64,64" \
                                                                --non_linearity $nl \
                                                                --traj_len $traj_len \
                                                                --init_cov $init_cov \
                                                                --free_nats $free_nats \
                                                                --measurement_uncertainty $measurement_uncertainty \
                                                                --weight_decay $weight_decay \
                                                                --use_stochastic_dynamics $use_stochastic_dynamics \
                                        done
                                    done
                                    wait
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
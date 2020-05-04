import argparse

from args.utils import str2bool, str2inttuple, str2tuple, str2floattuple

def parse_common_training_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Debug Settings
    parser.add_argument('--debug', type=str2bool, default=False,
                        help='Debug and do not save models or log anything')

    # Experiment Settings
    parser.add_argument('--storage_base_path', type=str, required=True,
                        help='Base path to store all training data')
    parser.add_argument('--resume_training_path', type=str, required=False, default='none',
                        help='Path to store previous weights, and optimizers to resume training')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for PyTorch')
    parser.add_argument('--cudnn_deterministic', type=str2bool, default=True,
                        help='Use cudnn deterministic')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False,
                        help='Use cudnn benchmark')
    parser.add_argument('--task', type=str, default="pendulum48",
                        help='The task that is being trained on')
    parser.add_argument('--comment', type=str, default="None",
                        help='Comment to describe model')

    # Dataset Settings
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Name of dataset to train on')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Amount of dataset to use for validation')

    # Training Settings
    parser.add_argument('--n_epoch', type=int, default=600,
                        help='Number of epochs',)
    parser.add_argument('--n_batch', type=int, default=128, 
                        help='Batch size')
    parser.add_argument('--n_example', type=int, default=10000000, 
                        help='Maximum samples to train from the dataset')
    parser.add_argument('--n_worker', type=int, default=4, 
                        help='Amount of workers for dataloading.')

    # Network Settings                        
    parser.add_argument('--use_batch_norm', type=str2bool, default=False,
                        help='Use batch normalization')
    parser.add_argument('--use_dropout', type=str2bool, default=False,
                        help='Use dropout')
    parser.add_argument('--weight_init', choices=['custom', 'none'], default='none', 
                        help='Weight initialization')

    # Optimizer Settings                        
    parser.add_argument('--beta1', type=float, default=0.9, 
                        help='Adam optimizer beta 1')
    parser.add_argument('--beta2', type=float, default=0.999, 
                        help='Adam optimizer beta 2')
    parser.add_argument('--opt', choices=['adam', 'sgd', 'adadelta', 'rmsprop'], default='adam', 
                        help='Optimizer used')
    parser.add_argument('--scheduler', choices=['none', 'exponential'], default='none', 
                        help='Scheduler used')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay rate')
    args = parser.parse_args()
    return args

def parse_training_args():
    parser = argparse.ArgumentParser()

    # Network args
    parser.add_argument('--dim_u', type=int, default=1,
                        help='Action dimension')
    parser.add_argument('--dim_a', type=int, default=3,
                        help='Emission state dimension')
    parser.add_argument('--dim_alpha', type=int, default=3,
                        help='Transition state dimension')
    parser.add_argument('--dim_z', type=int, default=3,
                        help='True state dimension')
    parser.add_argument('--dim_x', type=str2inttuple, default=(1, 48, 48),
                        help='3-tuple image dimension (C, H, W)')
    parser.add_argument('--k', type=int, default=2,
                        help='Number of dynamic models')
    parser.add_argument('--n_annealing_epoch_beta', type=int, default=0,
                        help='The number of annealing steps')
    parser.add_argument('--fc_hidden_size', type=int, default=50,
                        help='The number of hidden units for each linear layer')
    parser.add_argument('--alpha_hidden_size', type=int, default=55,
                        help='The number of hidden units for each GRU layer')
    parser.add_argument('--use_bidirectional', type=str2bool, default=False,
                        help='Use bidirectional RNN')
    parser.add_argument('--transition_noise', type=float, default=0.08,
                        help='Transition noise')
    parser.add_argument('--emission_noise', type=float, default=0.03,
                        help='Emission noise')
    parser.add_argument('--alpha_net', choices=['gru', 'lstm'], default='gru', 
                        help='Alpha network type')
    parser.add_argument('--measurement_net', choices=['fcn', 'cnn'], default='fcn',
                        help='Network architecture for measurement representation') 
    parser.add_argument('--non_linearity', choices=['relu', 'elu'], default='relu',
                        help='Activation used for neural network')
    parser.add_argument('--measurement_uncertainty', choices=['constant', 'scale', 'feature', 'learn_VAE', 'learn_separate', 'learn_separate_conc'], default='constant',
                        help='The type of measurement uncertainty used.')
    parser.add_argument('--use_stochastic_dynamics', type=str2bool, default=False,
                        help='Add some stochasticity to the dynamics.')

    # Training Settings
    parser.add_argument('--lr', type=float, default= 3e-4,
                        help='Learning rate')
    parser.add_argument('--opt_vae_epochs', type=int, default=0,
                        help='Number of epochs to train VAE only')
    parser.add_argument('--opt_vae_kf_epochs', type=int, default=10,
                        help='Number of epochs to train VAE and LGSSM (must be >= opt_vae_epochs)')
    parser.add_argument('--free_nats', type=float, default= 3.,
                        help='Amount of free nats allowed')
    parser.add_argument('--traj_len', type=int, default= 32,
                        help='Size of trajectory to train on')
    parser.add_argument('--init_cov', type=float, default= 40.,
                        help='Initial state covariance')

    # Loss Settings
    parser.add_argument('--lam_rec', type=float, default=1.0/256.0,
                        help='Weight of reconstruction loss')
    parser.add_argument('--lam_kl', type=float, default=1.0/256.0,
                        help='Weight of kl loss')
    parser.add_argument('--use_binary_ce', type=str2bool, default=False,
                        help='Use Binary Cross Entropy loss insted of default Mean Squared Error loss')

    args = parse_common_training_args(parser=parser)
    return args

def parse_tcp_server_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default='localhost',
                        help='Host Address')
    parser.add_argument('--port', type=int, default=5000,
                        help='Host Port')
    parser.add_argument('--device_id', type=int, default=0,
                        help='Camera Device ID')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera Resolution Height')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera Resolution Width')
    parser.add_argument('--frame_rate', type=int, default=30,
                        help='Camera Frame Rate')
    parser.add_argument('--colour_format', type=str, default='rgb8',
                        choices=['rgb8'], help='Camera Colour Format')

    args = parser.parse_args()
    return args

def parse_data_collection_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dbname', type=str, required=True,
                        help='Database file name to store')
    parser.add_argument('--args_output_file', type=str, required=True,
                        help='File name for storing arguments')

    parser.add_argument('--camera_res', type=str2inttuple, default=(3, 480, 640),
                        help='Camera Resolution')
    parser.add_argument('--hosts', type=str2tuple, default=('localhost', 'localhost'),
                        help='Hosts for connecting to RealSense cameras')
    parser.add_argument('--ports', type=str2inttuple, default=(5000, 5001),
                        help='Correspond ports for connecting to RealSense cameras')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')

    parser.add_argument('--dt', type=float, default=0.5,
                        help='Action frequency (in seconds)')
    parser.add_argument('--speed_max', type=float, default=0.5,
                        help='Maximum speed')
    parser.add_argument('--repeat_actions', type=int, default=3,
                        help='Number of times the same action is repeated until next action is sampled')
    parser.add_argument('--timeout', type=int, default=15,
                        help='The time limit of each episode')

    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to collect')

    parser.add_argument('--render', type=str2bool, default=False,
                        help='Visualize the cameras')

    args = parser.parse_args()
    return args

def parse_control_experiment_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dbname', type=str, required=True,
                        help='Database file name to store')
    parser.add_argument('--args_output_file', type=str, required=True,
                        help='File name for storing arguments')

    parser.add_argument('--model_path', type=str, required=True,
                        help='The path to the model')
    parser.add_argument('--device', type=str, required=False, default="cuda:0",
                        help='Device for the model')
    parser.add_argument('--goal_path', type=str, required=True,
                        help='The path specifying the goal')

    parser.add_argument('--T', type=int, default=4,
                        help='Number of history to initialize LGSSM')
    parser.add_argument('--mpc_horizon', type=int, default=5,
                        help='The number of actions MPC predicts')
    parser.add_argument('--enable_ood', type=str2bool, default=False,
                        help='Whether or not to enable OOD detection')

    parser.add_argument('--camera_res', type=str2inttuple, default=(3, 480, 640),
                        help='Camera Resolution')
    parser.add_argument('--hosts', type=str2tuple, default=('localhost', 'localhost'),
                        help='Hosts for connecting to RealSense cameras')
    parser.add_argument('--ports', type=str2inttuple, default=(5000, 5001),
                        help='Correspond ports for connecting to RealSense cameras')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')

    parser.add_argument('--dt', type=float, default=0.5,
                        help='Action frequency (in seconds)')
    parser.add_argument('--speed_max', type=float, default=0.5,
                        help='Maximum speed')
    parser.add_argument('--repeat_actions', type=int, default=3,
                        help='Number of times the same action is applied until next action is sampled')
    parser.add_argument('--timeout', type=int, default=15,
                        help='The time limit of each episode')

    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to collect')

    parser.add_argument('--render', type=str2bool, default=False,
                        help='Visualize the cameras')

    args = parser.parse_args()
    return args


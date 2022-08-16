import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--save_dir_path', default='train')
    parser.add_argument(
        '--load_model_path', default=None)
    parser.add_argument('--qualitative_value_study', action='store_true')
    parser.add_argument('--load_model_lvl', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train_load_model_path', default=None)
    parser.add_argument('--train_load_model_lvl', type=int, default=0)
    parser.add_argument('--model', default='XLVIN')
    parser.add_argument('--env_type', default='mountaincar')
    parser.add_argument('--transe_embedding_dim', type=int, default=50)
    parser.add_argument('--transe_hidden_dim', type=int, default=16)
    parser.add_argument('--env_action_dim', type=int, default=3)
    parser.add_argument('--env_input_dims', type=int, default=2)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--transe_encoder_type', default=None)
    parser.add_argument('--transe_pool_type', default=None)
    parser.add_argument('--transe_encoder_num_linear', type=int, default=0)
    parser.add_argument('--transe_vin_type', default=None)
    parser.add_argument('--pretrained_encoder', action='store_true')
    parser.add_argument('--transe_weights_path', default=None)
    parser.add_argument('--gnn', action='store_true')
    parser.add_argument('--gnn_hidden_dim', type=int, default=50)
    parser.add_argument('--message_function', type=str, default='mpnn')
    parser.add_argument('--message_function_depth', type=int, default=1)
    parser.add_argument('--neighbour_state_aggr', type=str, default='sum')
    parser.add_argument('--gnn_steps', type=int, default=2, help="How many GNN propagation steps are applied ")
    parser.add_argument('--msg_activation', action='store_true',
                        help='Whether to apply ReLU to messages')
    parser.add_argument('--gnn_layernorm', action='store_true', help='Whether to apply Layernorm to new representation')
    parser.add_argument('--gnn_weights_path', default=None)
    parser.add_argument('--num_processes', type=int, default=5)
    parser.add_argument('--cat_enc_gnn', action='store_true')
    parser.add_argument('--full_cat_gnn', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_gnn', action='store_true')
    parser.add_argument('--transe2gnn', type=int, default=0)
    parser.add_argument('--gnn_decoder', type=int, default=0)
    parser.add_argument('--vin_attention', action='store_true')
    parser.add_argument('--graph_detach', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--retrain_transe', action='store_true')
    parser.add_argument('--transe_loss_coef', type=float, default=0.001)
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--mini_batch_size',
        type=int,
        default=None,
        help='number of batches for ppo (default: None)')
    parser.add_argument('--transe_detach', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--num_train_episodes', type=int, default=5)
    parser.add_argument('--ppo_updates', type=int, default=1)
    parser.add_argument('--lvl_nb_deterministic_episodes', type=int, default=5)
    parser.add_argument('--lvl_threshold', type=float, default=None)

    args = parser.parse_args()

    return args
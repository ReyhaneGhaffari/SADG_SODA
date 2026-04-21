import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SADG-SODA: Style-Aware Density-Guided SF-OSDA')

    # Dataset settings
    parser.add_argument('--source', type=str, default='kather19',
                        choices=['kather16', 'kather19'],
                        help='Source dataset')
    parser.add_argument('--target', type=str, default='kather16',
                        choices=['kather16', 'kather19'],
                        help='Target dataset')
    parser.add_argument('--source_path', type=str,
                        default='/content/drive/MyDrive/SADG_SODA/data/kather19/NCT-CRC-HE-100K',
                        help='Path to source dataset')
    parser.add_argument('--target_path', type=str,
                        default='/content/drive/MyDrive/SADG_SODA/data/kather16/Kather_texture_2016_image_tiles_5000',
                        help='Path to target dataset')
    parser.add_argument('--split', type=int, default=1, choices=[1, 2, 3],
                        help='Closed/open set split configuration')

    parser.add_argument('--num_class', type=int, default=4,
                        help='Number of known (closed-set) classes')

    # Network settings
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--fc_hidden', type=list, default=[1024, 512],
                        help='FC layer hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate in FC layers')
    parser.add_argument('--rsam_ratio', type=float, default=0.5,
                        help='RSAM channel compression ratio')

    # Source training settings
    parser.add_argument('--source_epochs_k16', type=int, default=200,
                        help='Source training epochs for Kather-16')
    parser.add_argument('--source_epochs_k19', type=int, default=40,
                        help='Source training epochs for Kather-19')
    parser.add_argument('--source_lr', type=float, default=1e-3,
                        help='Source training learning rate')

    # Target adaptation settings
    parser.add_argument('--target_epochs_k16', type=int, default=20,
                        help='Target adaptation epochs for Kather-16')
    parser.add_argument('--target_epochs_k19', type=int, default=6,
                        help='Target adaptation epochs for Kather-19')
    parser.add_argument('--target_lr', type=float, default=1e-4,
                        help='Target adaptation learning rate')

    # Batch settings
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')

    # GMM settings
    parser.add_argument('--K', type=int, default=16,
                        help='Number of GMM components')
    parser.add_argument('--em_iterations', type=int, default=5,
                        help='Number of EM iterations per update')
    parser.add_argument('--purity_threshold', type=float, default=0.5,
                        help='Cluster purity threshold t')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='GMM prior refinement weight beta')

    # Loss weights
    parser.add_argument('--lambda_adv', type=float, default=0.1,
                        help='Adversarial loss weight')

    # Temperature scaling
    parser.add_argument('--tau_init', type=float, default=0.1,
                        help='Initial temperature parameter tau')

    # Feature modulation
    parser.add_argument('--epsilon_std', type=float, default=1.0,
                        help='Std of epsilon ~ N(0,1) for feature modulation')

    # GRL settings
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Max steps for GRL lambda scheduling')

    # High confidence selection
    parser.add_argument('--rho', type=float, default=0.5,
                        help='Rho parameter for threshold selection')

    # Image settings
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')

    # Device
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Logging
    parser.add_argument('--save_path', type=str,
                        default='/content/drive/MyDrive/SADG_SODA/checkpoints/',
                        help='Path to save model checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval in iterations')

    args = parser.parse_args(args=[])
    return args


if __name__ == '__main__':
    args = get_args()
    print("Settings loaded successfully:")
    print(f"  Source: {args.source} -> Target: {args.target}")
    print(f"  Split: {args.split}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Num classes: {args.num_class}")
    print(f"  K (GMM): {args.K}")
    print(f"  Device: {args.device}")
    print(f"  Source path: {args.source_path}")
    print(f"  Target path: {args.target_path}")
    print(f"  Save path: {args.save_path}")
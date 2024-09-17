import argparse


from data.Sample_dataset import MultiAudio
from model_vc import Generator
import os
from audioUtils.hparams import hparams


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  Creating %s...  ---" % path)
        print("---  OK  ---")

    else:
        print("---  %s already exists!  ---" % path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path1', nargs='+', required=True)
    parser.add_argument('--data_path2', nargs='+', required=True)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--dis', dest='dis', default=False, action='store_true')
    parser.add_argument('--use_lsgan', dest='use_lsgan', default=False, action='store_true')
    parser.add_argument('--lambda_gan', default=0.01, type=float)
    parser.add_argument('--multigpu', dest='multigpu', default=False, action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_freq', default=500, type=int)
    parser.add_argument('--display_freq', default=10, type=int)
    parser.add_argument('--lambda_wavenet', default=0.01, type=float)
    parser.add_argument('--test_path', default=None)
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--initial_iter', default=0, type=int)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--loss_content', dest='loss_content', default=False, action='store_true')

    args = parser.parse_args()

    print(args)

    dataloader_spk1 = MultiAudio(args.data_path1, batch_size=args.batch_size, num_workers=0)
    dataloader_spk2 = MultiAudio(args.data_path2, batch_size=args.batch_size, num_workers=0)
    if args.multigpu:
        device = 'cuda:0'
    else:
        device = args.device

    experimentName = args.experiment_name
    save_dir = os.path.join(args.save_dir, experimentName)
    mkdir("logs/" + experimentName)
    mkdir(save_dir)
    G = Generator(hparams.dim_neck, 512, hparams.freq, lr=1e-3, is_train=True,
                  loss_content=args.loss_content,
                  lambda_gan=args.lambda_gan,
                  multigpu=args.multigpu,
                  lambda_wavenet=args.lambda_wavenet,
                  test_path=args.test_path,
                  args=args).to(device)

    G.optimize_parameters(dataloader_spk1,dataloader_spk2, args.epochs, device, experimentName=experimentName, save_dir=save_dir,
                          save_freq=args.save_freq, display_freq=args.display_freq,
                          load_model=args.load_model,
                          initial_niter=args.initial_iter)




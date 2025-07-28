from dotenv import load_dotenv
load_dotenv()
import sys
if '/mnt/petrelfs/maokangkun/code/gits/SigmaFlow' not in sys.path:
    sys.path.append('/mnt/petrelfs/maokangkun/code/gits/SigmaFlow')
import argparse
# from src.base_pipeline_generator import gen_base_pipeline
from src.test import test
from src.exp import run_exp, run_exp_rl
from src.eval import eval_exp
# from src.auto import atuo_mode, analysis
from src.data import data_process
# from src.sft import sft, sft_test
# from src.grpo import grpo, grpo_test
# from src.server import run_server

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='run unit test')
    parser.add_argument('--exp', action='store_true', help='run experiment')
    parser.add_argument('--exp_rl', action='store_true', help='run experiment')
    parser.add_argument('--eval', action='store_true', help='eval experiment results')
    parser.add_argument('--gen', action='store_true', help='generate g0 pipeleine')
    parser.add_argument('--auto', action='store_true', help='generate g0 pipeleine')
    parser.add_argument('--ana', action='store_true', help='generate g0 pipeleine')
    parser.add_argument('--server', action='store_true', help='generate g0 pipeleine')
    parser.add_argument('--data', action='store_true', help='generate g0 pipeleine')
    parser.add_argument('--sft', action='store_true', help='generate g0 pipeleine')
    parser.add_argument('--sft_test', action='store_true', help='generate g0 pipeleine')
    parser.add_argument('--grpo', action='store_true', help='generate g0 pipeleine')
    parser.add_argument('--grpo_test', action='store_true', help='generate g0 pipeleine')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = setup_args()

    if args.test:
        test()
    elif args.exp:
        run_exp()
    elif args.exp_rl:
        run_exp_rl()
    elif args.eval:
        eval_exp()
    elif args.auto:
        atuo_mode()
    elif args.ana:
        analysis()
    elif args.server:
        run_server()
    elif args.data:
        data_process()
    elif args.sft:
        sft()
    elif args.sft_test:
        sft_test()
    elif args.grpo:
        grpo()
    elif args.grpo_test:
        grpo_test()
    elif args.gen:
        import os
        PIPELINE_DIR = os.getenv('PIPELINE_DIR', None)
        g0 = gen_base_pipeline(PIPELINE_DIR)
        print(g0)

if __name__ == '__main__':
    main()

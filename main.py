import os
from train import build_parser, train

os.environ['ATTN_BACKEND'] = 'xformers'


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
import argparse

parser = argparse.ArgumentParser()
# My arguments
parser.add_argument('--ds', type=int, default=0, help='dataset id')
parser.add_argument("--unsup", action="store_true", default=False, help="unsup")
parser.add_argument("--train_ratio", type=int, default=100, help="train_ratio")
parser.add_argument("--dual_no_time", action="store_true", default=False, help="use relation only")
parser.add_argument("--time_no_agg", action="store_true", default=False, help="Not aggregate time feature")
parser.add_argument("--sub_exps", action="store_true", default=False, help="test sim for every matrix")
parser.add_argument("--train_anyway", action="store_true", default=False, help="training over again")
parser.add_argument("--sep_eval", action="store_true", default=False, help="eval time sense/not sense separately")
parser.add_argument("--split", action="store_true", default=False, help="split the tkgs")

global_args = parser.parse_args()
print('IMPORTANT! current ARGs are', global_args)
unsup = global_args.unsup
which_file = global_args.ds
train_ratio = float(global_args.train_ratio) / 100
if which_file == 0:
    filename = 'data/ICEWS05-15/'
elif which_file == 1:
    filename = 'data/YAGO-WIKI50K/'
else:
    filename = 'data/YAGO-WIKI20K/'

import argparse


def str2list(v):
    return [int(d) for d in v.split(',')]


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


parser = argparse.ArgumentParser()
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset-name', type=str, default="cged")  # dataset name, control cache pkl name;
data_arg.add_argument('--dataset_dir', type=str, default="data")
data_arg.add_argument('--cache-file', type=str, default="")
data_arg.add_argument('--cache_data_directory', type=str, default="./cache/")
data_arg.add_argument('--generated_param_directory', type=str, default="./output")
data_arg.add_argument('--lm', type=str, default="./bart-base-chinese")

data_arg.add_argument('--theta', type=float, default=0.1)
data_arg.add_argument('--use-tlm', type=bool, default=False)
data_arg.add_argument('--masking-rate', type=float, default=0.9)
# data_arg.add_argument('--beta', type=float, default=0.3)
# data_arg.add_argument('--repetition_penalty', default=1.0, type=float, help='重复处罚率')
# data_arg.add_argument('--top_k', default=5, type=float, help='解码时保留概率最高的多少个标记')
# data_arg.add_argument('--weight_a', default=2, type=float)

learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--model_name', type=str, default="gec")
# mixed mode
learn_arg.add_argument('--threshold', type=float, default=0.5)
# 推理时也会用到
learn_arg.add_argument('--checkpoint', type=str, default="gecslaphappy-dandelion-uakari.pt")
learn_arg.add_argument('--batch-size', type=int, default=24)
learn_arg.add_argument('--valid-batch-size', type=int, default=24)
learn_arg.add_argument('--max-epoch', type=int, default=20)

learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=100)
learn_arg.add_argument('--print_steps', type=int, default=100)
learn_arg.add_argument('--valid-max-epoch', type=int, default=8)

learn_arg.add_argument('--lr', type=float, default=5e-5)
learn_arg.add_argument('--lr_decay', type=float, default=0.01)
learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
learn_arg.add_argument('--max_grad_norm', type=float, default=1.0)
learn_arg.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'])

evaluation_arg = add_argument_group('Evaluation')
evaluation_arg.add_argument('--max_length', type=int, default=500)
evaluation_arg.add_argument('--beam_search', type=int, default=6)
evaluation_arg.add_argument('--testing', type=str2bool, default=False)

misc_arg = add_argument_group('MISC')
misc_arg.add_argument('--refresh', type=str2bool, default=False)
misc_arg.add_argument('--device_ids', type=str2list, default='0')
misc_arg.add_argument('--visible_gpu', type=int, default=0)
misc_arg.add_argument('--random_seed', type=int, default=42)

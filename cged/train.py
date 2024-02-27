import os, torch
import random
import numpy as np
from utils.data import build_data
from trainer import Trainer
from datetime import datetime
from transformers import BertTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from config.base import get_args
import namegenerator
import pathlib


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args, unparsed = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.visible_gpu)  # 设置GPU卡号

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    set_seed(args.random_seed)

    print('loading data...')
    tokenizer = BertTokenizer.from_pretrained(args.lm)
    data = build_data(args)

    _name = namegenerator.gen()
    # 模型名称加时间戳，防止重复
    model_name = args.model_name + _name + datetime.now().strftime('%Y_%m_%d_%H_%M')
    # model name 避免重复
    while pathlib.Path(os.path.join(args.generated_param_directory, model_name + '.pt')).exists():
        model_name = args.model_name + namegenerator.gen() + datetime.now().strftime('%Y_%m_%d_%H_%M')

    args.model_name = model_name
    print(f'model_name: {args.model_name}')

    model = BartForConditionalGeneration.from_pretrained(args.lm, args)

    trainer = Trainer(model, data, args, device, tokenizer)

    if args.testing:
        dict = torch.load(os.path.join(args.generated_param_directory, args.checkpoint), map_location=device)
        model.load_state_dict(dict['state_dict'])
        model.eval()
        trainer.eval_model(data.test_loader, -1)
    else:
        trainer.train_model()

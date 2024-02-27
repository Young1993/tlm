# process cged to train and dev
import xml.dom.minidom
from tqdm import tqdm
import argparse
import sys
from os import path
import random

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import json
import opencc
from transformers import BertTokenizer

converter = opencc.OpenCC('t2s.json')


# 单独导出cged数据
def process_cged(args):
    xml_path = args.ori.split(',')

    lines_src, lines_tgt, lines_error = [], [], []
    # load CGED data
    for _path in xml_path:
        _lines_src, _lines_tgt, _errors = read_xml(_path)
        lines_src += _lines_src
        lines_tgt += _lines_tgt
        lines_error += _errors

    print(f"len of lines {len(lines_src)}")  # 10449

    sample, test_num = [], 0
    for src, tgt in zip(lines_src, lines_tgt):
        _src = tokenizer.encode(src, add_special_tokens=False)
        _tgt = tokenizer.encode(tgt, add_special_tokens=False)
        # if test_num > 4096:
        #     break
        # test_num += 1
        sample.append({
            "src": _src,
            "tgt": _tgt
        })

    random.shuffle(sample)
    _len = len(sample)
    _train_id = int(_len * 0.9)

    train_data = sample[:_train_id]
    dev_data = sample[_train_id:]

    with open(path.join(args.d, 'train.json'), 'w') as f_src:
        train_data = json.dumps(train_data, indent=2, ensure_ascii=False)
        f_src.write(train_data)
        f_src.close()

    with open(path.join(args.d, 'dev.json'), 'w') as f_tgt:
        dev_data = json.dumps(dev_data, indent=2, ensure_ascii=False)
        f_tgt.write(dev_data)
        f_tgt.close()

    return


def read_xml(path):
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    print(root.nodeName)

    lines_src, lines_tgt, errors = [], [], []
    for ele in tqdm(root.childNodes, desc="extract text"):
        if ele.nodeName == 'DOC':
            sample = {'errors': []}
            for child_ele in ele.childNodes:
                if child_ele.nodeName == 'TEXT':
                    sample['src'] = child_ele.childNodes.item(0).data.strip()
                if child_ele.nodeName == 'CORRECTION':
                    sample['tgt'] = child_ele.childNodes.item(0).data.strip()
                if child_ele.nodeName == 'ERROR':
                    # sample['errors'].append(child_ele.attributes.items())
                    sample['errors'].append([o[1] for o in child_ele.attributes.items()])

            assert 'src' in sample and 'tgt' in sample
            if not len(sample['errors']):
                sample['errors'].append(['correct'])

            if len(sample['src']) >= 510 or not len(sample['src']) or len(sample['tgt']) >= 510:
                # len(sample['src']) >= len(sample['tgt']) + 10 or len(sample['tgt']) >= len(
                # sample['src']) + 10:
                print(f"{sample['src']}")
                print(f"{sample['tgt']}")
                continue

            lines_src.append(converter.convert(sample['src']))
            lines_tgt.append(converter.convert(sample['tgt']))
            errors.append(sample['errors'])
        else:
            # if ele.nodeName != '#text':
            # print(ele.nodeName)
            for child_ele in ele.childNodes:
                # if child_ele.nodeName == 'TEXT':
                print(child_ele.childNodes.item(0).data.strip())

    return lines_src, lines_tgt, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', default="../data/cged2017_src.txt", type=str, help="语法输入文件")
    parser.add_argument('-t', default="../data/cged2017_tgt.txt", type=str, help="语法输出文件")
    parser.add_argument('-ori', default="../data/train.release.xml", type=str, help="源文件")
    parser.add_argument('-d', default="../data", type=str, help="数据集存放目录")

    args, _ = parser.parse_known_args()
    tokenizer = BertTokenizer.from_pretrained("../bart-base-chinese")
    process_cged(args)

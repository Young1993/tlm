#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os


if __name__ == '__main__':
    # file = './glue_out/tmp_base_rte/'
    # output_file = './submit_tmp_base_rte/'
    file = './glue_out/tmp_tlm_rte/'
    output_file = './submit_tmp_tlm_rte/'
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    map_dic = {
        'predict_results_cola.txt': 'CoLA.tsv',
        'predict_results_mrpc.txt': 'MRPC.tsv',
        'predict_results_mnli.txt': 'MNLI-m.tsv',
        'predict_results_mnli-mm.txt': 'MNLI-mm.tsv',
        'predict_results_qnli.txt': 'QNLI.tsv',
        'predict_results_qqp.txt': 'QQP.tsv',
        'predict_results_rte.txt': 'RTE.tsv',
        'predict_results_sst2.txt': 'SST-2.tsv',
        'predict_results_stsb.txt': 'STS-B.tsv',
        'predict_results_wnli.txt': 'WNLI.tsv',
    }

    for filename in os.listdir(file):
        new_file = file + filename
        for k in map_dic:
            if k in os.listdir(new_file):
                ori_name = new_file + '/' + k
                os.environ['ori_name'] = str(ori_name)
                os.environ['output_file'] = str(output_file)
                os.system('cp $ori_name $output_file')

    for filename in os.listdir(output_file):
        if filename in map_dic:
            ori_name = output_file + filename
            if filename in map_dic:
                new_name = output_file + map_dic[filename]
                os.environ['ori_name'] = str(ori_name)
                os.environ['new_name'] = str(new_name)
                os.system('mv $ori_name $new_name')


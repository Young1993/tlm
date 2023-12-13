# TLM: Token-Level Masking for Transformers (EMNLP2023)
The public code.\
In this paper, we propose a new regularization scheme based on token-level rather than structure-level to reduce overfitting. Specifically, we devise a novel Token-Level Masking (TLM) training strategy for Transformers to regularize the connections of self-attention, which consists of two masking techniques that are effective and easy to implement. The underlying idea is to manipulate the connections between tokens in the multi-head attention via masking, where the networks are forced to exploit partial neighbors’ information to produce a meaningful representation. 

## Note
we release the code => BERT-base (bert-base-cased) with TLM on RTE

## Installation
```shell
git clone https://github.com/Young1993/tlm.git
cd tlm/
pip install -r requirements.txt
```

### Use TLM
```shell
cd glue
sh bert-base_sh/rte_train_tlm.sh
```
### without TLM
cd glue
sh bert-base_sh/rte_train.sh

### Notice

# Todo
- [x] Run Bert-base with TLM
- [ ] To test QWen/LLama/sqlcoder with TLM


#Citation
```text
@inproceedings{wu-etal-2023-tlm,
    title = "{TLM}: Token-Level Masking for Transformers",
    author = "Wu, Yangjun  and
      Fang, Kebin  and
      Zhang, Dongxiang  and
      Wang, Han  and
      Zhang, Hao  and
      Chen, Gang",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.871",
    pages = "14099--14111"
}
```

[TLM: Token-Level Masking for Transformers](https://aclanthology.org/2023.emnlp-main.871) (Wu et al., EMNLP 2023)

If you have any questions, please feel free to talk with me (email: <em>yangjun.wu@connect.polyu.hk<em>)
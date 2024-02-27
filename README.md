# TLM: Token-Level Masking for Transformers (EMNLP2023)

The public code.\
In this paper, we propose a new regularization scheme based on token-level rather than structure-level to reduce
overfitting. Specifically, we devise a novel Token-Level Masking (TLM) training strategy for Transformers to regularize
the connections of self-attention, which consists of two masking techniques that are effective and easy to implement.
The underlying idea is to manipulate the connections between tokens in the multi-head attention via masking, where the
networks are forced to exploit partial neighbors’ information to produce a meaningful representation.

## <em>Note</em>

we release the code => BERT-base (bert-base-cased) with TLM on RTE

- if you do not use attention dropout, set **attention_probs_dropout_prob=0**
- if you do not use tlm, set **--use_tlm=0**

## 1 Installation

```shell
git clone https://github.com/Young1993/tlm.git
cd tlm/
pip install -r requirements.txt
```

### 1.1 Training with TLM based on Bert

```shell
cd glue
sh bert-base_sh/rte_train_tlm.sh
```

### 1.2 Training without TLM based on Bert

cd glue sh bert-base_sh/rte_train.sh

### 🆕1.3 Training with TLM based on Bart
1. First, you need to install fnlp/bart-base-chinese;
2. Second, run preprocess.py
3. The code is not high efficient, I will update it soon.
```shell
cd cged
sh train_tlm.sh 0
```

### Notice

If you plan to apply tlm to decoder-only architecture, you can reference to the code in
./transformers/models/bert/modeling_bert.py.

# Todo

- ✅ Run Bert-base with TLM.
- Run Bart-base with TLM
- To test QWen/LLama/sqlcoder with TLM.
- To pull requests code into Transformers.

# Citation

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
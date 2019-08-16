# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/30 16:47
 @Author  : MaCan (ma_cancan@163.com)
 @File    : __init__.py.py
"""


def start_server():
    from bert_base.server import BertServer
    from bert_base.server.helper import get_run_args

    args = get_run_args()
    # print(args)
    server = BertServer(args)
    server.start()
    server.join()


def start_client():
    pass


def train_ner():
    import os
    from bert_base.train.train_helper import get_args_parser
    from bert_base.train.bert_lstm_ner import train

    args = get_args_parser()
    # args.do_eval=False
    # args.do_train=False
    # args.do_predict=True
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    train(args=args)


if __name__ == '__main__':
     # start_server()
     '''
       -do_predict True -do_eval False -do_train False -data_dir /Users/soledede/sourcecode/nlp/ner/Name-Entity-Recognition/BERT-BiLSTM-CRF-NER/NERdata -output_dir /Users/soledede/sourcecode/nlp/ner/Name-Entity-Recognition/BERT-BiLSTM-CRF-NER/output/ner_result -init_checkpoint /Users/soledede/Documents/model/BERT/chinese_L-12_H-768_A-12/bert_model.ckpt -bert_config_file /Users/soledede/Documents/model/BERT/chinese_L-12_H-768_A-12/bert_config.json  -vocab_file /Users/soledede/Documents/model/BERT/chinese_L-12_H-768_A-12/vocab.txt -label_list 'O,B-PER,I-PER,B-ORG,I-ORG,B-LOC,I-LOC,X,[CLS],[SEP]'
     '''
     '''
     -do_predict True -do_eval False -do_train False -data_dir /Users/soledede/sourcecode/nlp/ner/Name-Entity-Recognition/BERT-BiLSTM-CRF-NER/tourism_data -output_dir /Users/soledede/sourcecode/nlp/ner/Name-Entity-Recognition/BERT-BiLSTM-CRF-NER/output/tourism  -init_checkpoint /Users/soledede/Documents/model/BERT/chinese_L-12_H-768_A-12/bert_model.ckpt -bert_config_file /Users/soledede/Documents/model/BERT/chinese_L-12_H-768_A-12/bert_config.json  -vocab_file /Users/soledede/Documents/model/BERT/chinese_L-12_H-768_A-12/vocab.txt -label_list 'B-PER,I-PER,O,B-SCE,I-SCE,B-TIM,I-TIM,B-HOT,I-HOT,B-RES,I-RES,B-DLO,I-DLO,B-DIS,I-DIS,B-ORG,I-ORG,B-TEL,I-TEL,B-PRI,I-PRI,B-TIC,I-TIC,X,[CLS],[SEP]'
     '''
     train_ner()
# encoding=utf-8

"""
基于命令行的在线预测方法
@Author: Macan (ma_cancan@163.com) 
"""
import pandas as pd
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
import csv
from datetime import datetime

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
import  re
from decimal import Decimal

args = get_args_parser()
model_dir = r"C:\Users\cn190312\Documents\NPL_Code\Bert\bert-bilstm-crf-ner\output\npl_m"
bert_dir = r'C:\Users\cn190312\Documents\NPL_Code\Bert\chinese_L-12_H-768_A-12'


is_training=False
use_one_hot_embeddings=False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


class Tokenizer(object):
    def tokenize(self,text):
        raise NotImplementedError()


class SinglerTokenizer(Tokenizer):
    def tokenize(self,text,covert_num=False):

       text = self.clean(text, covert_num)
       tokens = []
       if(covert_num):
           blocks = text.split('<M>')
           token_list = []
           for block in blocks:
               token_list.append(list(block))
               token_list.append(['<M>'])
           del token_list[len(token_list)-1]
           tokens = [token for elem in token_list for token in elem]
       else:
           tokens = [token for token in list(text) if token != '\ufeff']
       return tokens

    def clean(self,text,covert_num=False):
        # 去掉空格
        text = text.replace(' ', '')
        # 多个换行符转换为一个
        text = re.sub(re.compile(r'\n+', re.S), '\n', text)
        # 替换掉数字
        if covert_num:
         text = re.sub(re.compile(r'[0-9]+\.?[0-9]*', re.S), '<M>', text)
        return text




tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)



tokenizer_my =  SinglerTokenizer()

def predict_online():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, args.max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, args.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        data = pd.read_csv("C:/Users/cn190312/Documents/NPL_Code/initial.csv", encoding='utf-8', error_bad_lines=False)
        # data = pd.read_csv("C:/Users/cn190312/Documents/分句/hehe2.csv", encoding='utf-8', error_bad_lines=False)
        address = data['抵押物明细'].values.tolist()
        print(id2label)
        #if True:
        result_list = []
        for i in address:
            i = i.replace('\r','').replace('\n','').replace('\t','')
            sentence=str(i)

            convert_sentences_index = []
            list_number = re.finditer(r'[0-9]+\.?[0-9]*', sentence)
            for num in list_number:
                start,end = num.span()
                convert_sentences_index.append((start,end))
            origin_sentence = sentence
            convert_m_sentences_index = []
            #替换数字为<M>
            sentence = re.sub(re.compile(r'[0-9]+\.?[0-9]*', re.S), 's', sentence)

            print('converted sentence:', sentence)
            sentece_m_num = re.finditer(r's', sentence,re.IGNORECASE)
            for num in sentece_m_num:
                start, end = num.span()
                convert_m_sentences_index.append((start, end))

            m_num_dic = {}
            for point1,point2 in zip(convert_m_sentences_index,convert_sentences_index):
                m_num_dic[point1] = point2

            start = datetime.now()
            if len(sentence) < 2:
                print(sentence)
                continue
            sentence = tokenizer.tokenize(sentence)
            #sentence = tokenizer_my.tokenize(text = sentence,covert_num=True)

            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask}
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            print(pred_label_result)

            #结构化输出
            sentence_list = struct_sentence(sentence,pred_label_result[0],origin_sentence,m_num_dic)
            if '商服' in sentence_list[2]:
                sentence_list[2] = '商服用地'
            elif '商住' in sentence_list[1]:
                sentence_list[2] = '商住用地'
            elif '商' in sentence_list[1]:
                sentence_list[2] = '商业用地'
            elif '工' in sentence_list[1]:
                sentence_list[2] = '工业用地'
            elif '住宅'  or '别墅' in sentence_list[1]:
                sentence_list[2] = '住宅'
            elif '厂' in sentence_list[1]:
                sentence_list[2] = '厂房'
            elif '写字'or '办公' in sentence_list[1]:
                sentence_list[2] = '办公'
            else:
                sentence_list[2] = sentence_list[1]

            result_list.append(sentence_list)
        print(result_list)

        with open("C:/Users/cn190312/Documents/NPL_Code/NER.csv", "w",encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Origin_sentence','Address','type','Building_Area','Land_Area'])
            writer.writerows(result_list)

        # output = open(r'C:\Users\cn190312\Documents\NPL_Code\NER.csv', 'w', encoding='utf-8')
        # # output.write('\nOrigin_sentence\tAddress\ttype\tBuilding_Area\tLand_Area\n')
        # for i in range(len(result_list)):
        #     for j in range(len(result_list[i])):
        #         output.write(str(result_list[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
        #         output.write('\t')  # 相当于Tab一下，换一个单元格
        #     output.write('\n')  # 写完一行立马换行
        # output.close()

def struct_sentence(tokens,labels,origin_sentence,m_num_dic):
    '''
    
    :param tokens:  输入句子的token列表 type:list()
    :param labels:  预测的标注列表 type:list
    :param origin_sentence: 原始句子
    :param m_num_dic: m number origin sentence mapping, (28,31)->(29,38)
    :return:  Collateral
    '''
    result = []
    col = Collateral()
    start = -1
    end = -1
    index = -1
    pre_label = ''
    for label in labels:
        index += 1

        if ((label in ['O'] or (label.find("B-") != -1 and pre_label.find("I-") != -1 and pre_label != ''))  and start != -1) :
            end = index

            if(True):
                    token_sub = tokens[start:end ]
                    labels_sub = labels[start:end]

                    token_start_sub = tokens[0:end]

                    token_start_value = ''.join(token_start_sub)

                    token_start_value_list = []

                    for i,token in enumerate(token_start_value):
                            if (i,i+1) in m_num_dic:
                                o_start, o_end = m_num_dic[(i, i+1)]
                                token_start_value_list.append(origin_sentence[o_start:o_end])
                            else:
                                token_start_value_list.append(token)

                    token_sub = token_start_value_list[start:]

                    token_value =''.join(token_sub)
                    label_value =''.join(labels_sub)

                    #print(label_value, token_value)

                    start = -1
                    end = -1
                    if 'Collateral_Address' in label_value:
                        col.Collateral_Address = token_value
                    elif 'Collateral_Type' in label_value:
                        col.Collateral_Type = token_value
                    elif 'Building_Area' in label_value:
                        number = re.findall(r"\d+\.?\d*", token_value)
                        if number != '' and len(number) > 0:
                            number = float(number[0])
                            if  '亩' in token_value:
                                number =  number* 666.6666667
                                number = Decimal(number).quantize(Decimal('0.00'))
                        col.Building_Area = number
                    elif 'Land_Area' in label_value:
                        number = re.findall(r"\d+\.?\d*", token_value)
                        if number != '' and len(number) > 0:
                            number = float(number[0])
                            if '亩' in token_value:
                                number = number * 666.6666667
                                number = Decimal(number).quantize(Decimal('0.00'))
                        col.Land_Area = number
        if label.find("B-") != -1:
            start  = index
        pre_label = label

    if start != -1 and index == len(labels)-1:
        end = index
        token_sub = tokens[start:end]
        labels_sub = labels[start:end]

        token_start_sub = tokens[0:end]

        token_start_value = ''.join(token_start_sub)

        token_start_value_list = []

        for i, token in enumerate(token_start_value):
            if (i, i + 1) in m_num_dic:
                o_start, o_end = m_num_dic[(i, i + 1)]
                token_start_value_list.append(origin_sentence[o_start:o_end])
            else:
                token_start_value_list.append(token)

            token_sub = token_start_value_list[start:]

            token_value = ''.join(token_sub)
            label_value = ''.join(labels_sub)

            if 'Collateral_Address' in label_value:
                col.Collateral_Address = token_value
            elif 'Collateral_Type' in label_value:
                col.Collateral_Type = token_value
            elif 'Building_Area' in label_value:
                number = re.findall(r"\d+\.?\d*", token_value)
                if number != '' and len(number) > 0:
                    number = float(number[0])
                    if '亩' in token_value:
                        number = number * 666.6666667
                        number = Decimal(number).quantize(Decimal('0.00'))
                col.Building_Area = number
            elif 'Land_Area' in label_value:
                number = re.findall(r"\d+\.?\d*", token_value)
                if number != '' and len(number) > 0:
                    number = float(number[0])
                    if '亩' in token_value:
                        number = number * 666.6666667
                        number = Decimal(number).quantize(Decimal('0.00'))
                col.Land_Area = number

    li = [origin_sentence,col.Collateral_Address,col.Collateral_Type,col.Building_Area,col.Land_Area]
    return li
    #li = col.print_out()
    # list_csv(li)



#out = open('C:/Users/cn190312/Documents/sao.csv', 'a',newline='')
class Collateral():
   def __init__(self,Collateral_Address='',Collateral_Type='',Building_Area='',Land_Area=''):
       self.Collateral_Address = Collateral_Address
       self.Collateral_Type = Collateral_Type
       self.Building_Area = Building_Area
       self.Land_Area =Land_Area

   def print_out(self):
        # print("Collateral_Address:",self.Collateral_Address)
        # print("Collateral_Type:",self.Collateral_Type)
        # print("Building_Area:",self.Building_Area)
        # print("Land_Area:",self.Land_Area)
        # print("\n")
        # out = open('C:/Users/cn190312/Documents/sao.csv', 'w', newline='')
        Li = [self.Land_Area,self.Building_Area,str(self.Collateral_Type),str(self.Collateral_Address)]
        print(type(Li))


def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result



def strage_combined_link_org_loc(tokens, tags):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """
    def print_output(data, type):
        line = []
        line.append(type)
        for i in data:
            line.append(i.word)
        print(', '.join(line))

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    person, loc, org = eval.get_result(tokens, tags)
    print_output(loc, 'LOC')
    print_output(person, 'PER')
    print_output(org, 'ORG')


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []
    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))


if __name__ == "__main__":
    predict_online()
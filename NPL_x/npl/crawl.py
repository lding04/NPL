from npl.baiduApi import bd_geo_encode
import re
import pandas as pd
import os
import requests
import argparse
import configparser
from bs4 import BeautifulSoup
import json
import csv
from npl import settings, conf
from pathlib import Path

base_folder = os.path.dirname(__file__)


class AuctionItem(object):
    def __init__(self, json_data):
        self.item_id = json_data['id']
        self.status = json_data['status']
        self.img_url = json_data['picUrl']
        self.item_url = conf.get('url', 'item_endpoint').format(self.item_id)
        self.synonym = configparser.ConfigParser(allow_no_value=True)
        self.synonym.read(settings.DICT_FILE, encoding='utf8')
        self.soup = BeautifulSoup(requests.get(self.item_url).content, 'html.parser')
        tag = self.soup.find('div', id='J_desc')
        self.table = BeautifulSoup(requests.get('https:' + tag['data-from']).text.strip()[10: -2], 'html.parser')
        self.title_text = self.soup.find('div', class_='pm-main').find('h1').text.strip()
        self.notice_url = 'https:' + self.soup.find('a', class_='item-announcement')['href'].replace('susong', 'sf') \
            .replace('court_notice', 'notice_detail')
        soup = BeautifulSoup(requests.get(self.notice_url).content, 'html.parser', from_encoding='gbk')
        self.notice = soup.find('div', class_='notice-content').text.strip()
        self.caution = '未找到'
        self.fully_match = '未确认生成逻辑'
        self.others = 'N/A'

    @property
    def title(self):
        return self.soup.find('title').text[:-14].strip()

    @property
    def auction_time(self):
        s = re.findall(r'\w+', self.title_text)[0]
        for key, value in {'一': 1, '二': 2, '三': 3}.items():
            if key in s:
                return value
        return '未找到'

    @property
    def address(self):
        s = self.soup.find(class_='item-address').text.replace(' ', '').strip()
        return self.title if len(s) < 10 else s

    @property
    def auction_type(self):
        element = self.soup.find('table', class_='pai-pay-infor').find('span', class_='pay-type')
        return element.text.strip() if element else '未找到'

    @property
    def consult_price(self):
        element = (self.soup.find('span', class_='pay-mark i-b', text='评 估 价').find_next('span', class_='J_Price')
                   or self.soup.find('span', class_='pay-mark i-b', text='市 场 价').find_next('span', class_='J_Price'))
        return element.text.replace(',', '').strip() if element else '未找到'

    @property
    def init_price(self):
        element = self.soup.find('span', class_='pay-mark i-b', text='起 拍 价').find_next('span', class_='J_Price')
        return element.text.replace(',', '').strip() if element else '未找到'

    @property
    def current_price(self):
        if self.status == 'failure':
            return 'N/A'
        element = self.soup.find('span', class_='pm-current-price J_Price')
        return element.text.replace(',', '').strip() if element else '未找到'

    @property
    def start_time(self):
        res = re.findall(r'于(.*?时)', self.notice)
        return res[0] if res else '未找到'

    @property
    def end_time(self):
        res = re.findall(r'至(.*?时)', self.notice)
        return res[0] if res else '未找到'

    @property
    def attachment(self):
        resp = requests.get(conf.get('url', 'item_attachment'), params={'id': self.item_id})
        attachment_id_list_data = json.loads(resp.text.strip()[5: -2])
        attachments = {d['title'].replace('\\', ''): conf.get('url', 'item_attachment_download').format(d['id']) for d
                       in
                       attachment_id_list_data}
        return attachments if attachments else '无附件'

    @property
    def disposal_unit(self):
        element = self.soup.find('span', class_='unit-txt unit-name item-announcement')
        return element.text.strip()

    @property
    def land_usage(self):
        element = self.table.find('span', text=re.compile(r'用地')) or self.table.find(text='土地用途').find_next('td')
        return element.text.strip() if element else 'N/A'

    @property
    def land_area(self):
        return self._find_by_regex('landArea', r'[是约为合共计:：]{0,5}(\d+(\.\d+)?)')

    @property
    def built_area(self):
        return self._find_by_regex('builtArea', r'[是约为合共计:：]{0,5}(\d+(\.\d+)?)')

    @property
    def land_cert(self):
        return self._find_by_regex('landCertificate', r'[是约为合共计:：]{0,5}(.*?号)')

    @property
    def coordinate(self):
        return str(bd_geo_encode(self.address))

    @property
    def land_termination_date(self):
        s = re.findall(r'终止日期[是约为合共计:：]{0,5}(.*?日)', self.notice)
        return s[0] if s else 'N/A'

    @property
    def house_cert(self):
        return self._find_by_regex('houseCertificate', r'[是约为合计:：]{0,5}(.*?号)')

    @property
    def land_nature(self):
        nature = ['综合用地', '居住用地', '工业用地', '教育文化用地', '商业用地']
        for n in nature:
            if n in self.table.text:
                return n
        element = self.table.find(text='土地性质').find_next('td')
        return element.text if element else 'N/A'

    @property
    def house_usage(self):
        element = self.table.find(text='房屋用途').find_next('td')
        return element.text.strip() if element else 'N/A'

    def _find_by_regex(self, keyword, expr):
        return (self._find_by_regex_content(keyword, expr, self.notice) or
                self._find_by_regex_content(keyword, expr, self.table.text) or '未找到')

    def _find_by_regex_content(self, keyword, expr, content):
        for synonym in self.synonym[keyword]:
            pattern = synonym + expr
            founds = re.findall(pattern, content)
            if len(founds) == 1:
                if isinstance(founds[0], tuple):
                    return founds[0][0]
                return founds[0]
            elif len(founds) > 1:
                return '找到多个'


class Target(object):
    def __init__(self, target_id,  keyword):
        self.target_id = target_id
        self.keyword = keyword
        self.auction_items = []
        self.field_mapping = {}
        for row in pd.read_csv(settings.COLUMN_FILE).to_dict('records'):
            self.field_mapping[row['field']] = row['name']
        self._item_df = None
        self.add_items()
        self.coordinate = str(bd_geo_encode(self.keyword))

    def add_items(self, url=None, max_page=10):
        url = conf.get('url', 'search_keyword') if url is None else url
        for page in range(1, max_page):
            query = {'q': self.keyword.encode('gbk'), 'spm': 'a213w.3064813.9001.1', 'page': page}
            soup = BeautifulSoup(requests.get(url, params=query).text, 'html.parser')
            data = json.loads(soup.find(id='sf-item-list-data').text)['data']
            if not data:
                break
            for d in data:
                item = AuctionItem(d)
                self.auction_items.append(item)

    @property
    def items_df(self):
        column_list = self.field_mapping.values()
        df = pd.DataFrame(columns=column_list)
        if not self.auction_items:
            df.append(pd.DataFrame())
        for item in self.auction_items:
            item_dict = {v: [getattr(item, k)] if hasattr(item, k) else ['N/A'] for k, v in self.field_mapping.items()}
            df = df.append(pd.DataFrame(item_dict))
        df['目标ID'] = self.target_id
        df['目标名称'] = self.keyword
        df['目标坐标'] = self.coordinate
        return df

    def output(self, output_path):
        if self.auction_items:
            output_file = Path(output_path).resolve() / (self.target_id+'.xlsx')
            print('  writing result to file {}'.format(output_file))
            self.items_df.to_excel(output_file, index=False)


class Crawler(object):
    def __init__(self, arg):
        self.input_file = arg.input_file
        self.output_path = arg.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.merge = arg.merge

    def crawl(self):
        df = pd.read_csv(self.input_file)
        dfs = []
        for index, row in df.iterrows():
            _target = Target(row['id'], row[4])
            print('({}/{})Processing...{}'.format(index+1, df.count()['id'], _target.keyword))
            print(' {} items found.'.format(len(_target.auction_items)))
            i = 5
            while isinstance(row[i],float) is False:
                _target = Target(row['id'], row[i])

                if len(_target.auction_items) == 0:
                    _target = Target(row['id'], row[i])
                    print('({}/{})Processing...{}'.format(index + 1, df.count()['id'], _target.keyword))
                    print(' {} items found.'.format(len(_target.auction_items)))
                    _target.add_items(max_page=conf.getint('core', 'max_page'))
                    i=i+1
                else:
                    print('({}/{})Processing...{}'.format(index + 1, df.count()['id'], _target.keyword))
                    print(' {} items found.'.format(len(_target.auction_items)))
                    break
            _target.output(self.output_path)
            dfs.append(_target.items_df)
        if self.merge:
            df = pd.concat(dfs, sort=False)
            # df.to_csv('result.csv', index=False, quoting=csv.QUOTE_ALL)
            output_file = Path(self.output_path).resolve() / 'result.xlsx'
            print('  writing final result to file {}'.format(output_file))
            df.to_excel(output_file, index=False)


parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', nargs='?', dest='input_file', help='a csv format file, contains (id, keyword)',
                    default=settings.INPUT_FILE)
parser.add_argument('--output', '-o', nargs='?', dest='output_path', help='the path you want to dump the result',
                    default=settings.OUTPUT_PATH)
parser.add_argument('--download', '-D', action='store_false', dest='download',
                    help='whether or not download attachments')
parser.add_argument('--merge', '-m', action='store_false', dest='merge',
                    help='whether or not merge results')

if __name__ == '__main__':
    args = parser.parse_args()
    crawler = Crawler(args)
    crawler.crawl()

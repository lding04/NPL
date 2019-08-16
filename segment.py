import pandas as pd
import cpca
import re

raw_data = pd.read_csv("NER.csv", encoding='utf-8')
addresses = list(raw_data["Address"][5:])

address_lists = []
for address in addresses:
    address_norm = cpca.transform([address]).values.tolist()[0]
    cutting = re.findall(r'\D+|\d+', str(address_norm[-1]))
    address_list = []
    for num, strings in enumerate(cutting):
        if strings.isdigit():
            address_list.append(''.join(cutting[0:num]))
    address_list.reverse()
    address_norm.extend(address_list)
    address_lists.append(address_norm)

address_lists = pd.DataFrame(address_lists).to_csv('Segment_Final.csv', encoding='utf-8')

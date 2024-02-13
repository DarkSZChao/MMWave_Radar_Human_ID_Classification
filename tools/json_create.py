import json

data = {
    'Cindy': 0,
    'HCX'  : 1,
    'Kaan' : 2,
    'Leo'  : 3,
    'LMX'  : 4,
    'LZ'   : 5,
    'Naim' : 6,
    'Piggy': 7,
    'Qi'   : 8,
    'Sandy': 9,
    'SZC'  : 10,
    'XA'   : 11,
    'XSJ'  : 12,
    'YJR'  : 13,
    'ZCH'  : 14,
    'ZCN'  : 15,
    'ZQY'  : 16,
}

json_data = json.dumps(data, indent=2)  # indent参数用于指定缩进空格数，可选

with open('data.json', 'w') as json_file:
    json_file.write(json_data)

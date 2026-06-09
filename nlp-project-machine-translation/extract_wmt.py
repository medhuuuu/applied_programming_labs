import json

with open('data/wmt24pp_test.jsonl', 'r') as f, \
     open('data/test.en', 'w') as out_en, \
     open('data/test.sv', 'w') as out_sv:
    for line in f:
        data = json.loads(line)
        out_en.write(data['source'] + '\n')
        out_sv.write(data['target'] + '\n')

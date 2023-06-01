import json

import pandas as pd

with open('alpaca_data.json') as f:
    data = json.load(f)


new_format = []
for i, point in enumerate(data):
    # no input
    if len(point['input']) == 0:
        inputt = "Below is an instruction that describes a task.\n "
        inputt += "Write a response that appropriately completes the request.\n\n"
        inputt += f"### Instruction:\n{point['instruction']}\n\n### Response:"
    else:
        inputt = "Below is an instruction that describes a task.\n "
        inputt += "Write a response that appropriately completes the request.\n\n"
        inputt += f"### Instruction:\n{point['instruction']}\n\n### Input:\n{point['input']}\n\n### Response:"

    item = {'input': inputt, 'output': str(point['output'])}
    new_format.append(item)

df = pd.DataFrame(new_format)
df = df.dropna()
df.to_csv('alpaca_data.csv')

import json
import sys

path_1, path_2 = sys.argv[1], sys.argv[2]

with open(path_1, "r") as f:
    json_data_1 = json.load(f)

with open(path_2, "r") as f:
    json_data_2 = json.load(f)

json_data = json_data_1 + json_data_2

print(f'finally {len(json_data)} datas')
with open('combined.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

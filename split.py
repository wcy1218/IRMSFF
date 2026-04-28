import pandas as pd
import re

df = pd.read_csv("data/source.csv", header=None)
code = df[0].tolist()
result = []

pattern = r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])'

for i in range(len(code)):
    tmp = code[i]
    words = re.split(pattern, tmp)
    result.append(' '.join([w for w in words if w]))

df = pd.DataFrame(result)
df.to_csv("target.csv", index=False, header=None)

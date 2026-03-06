import pandas as pd
import json

all_data = []

txt_df = pd.read_csv('all.txt', sep='\t', on_bad_lines='skip')
for index, row in txt_df.iterrows():
    words = eval(row.iloc[1])
    labels = eval(row.iloc[4])
    punc = "\"\'.,:?;"
    contraction = "n\'t"
    comment = ""
    current_label = 'non'
    for i in range(len(words)):
        if labels[i] == 'none':
            continue

        current_label = labels[i]
        if (words[i] != contraction and
            words[i][0] not in punc
            and i > 0 and words[i-1] != ('\'' or "\"")):
            comment += " "
        comment += words[i]

    if comment:
        print(comment + "\n" + current_label.upper() + "\n")
        all_data.append([comment, current_label])

# TODO: read json file and append to all_data

all_df = pd.DataFrame(all_data, columns=["text", "labels"])
print(all_df.tail(2))

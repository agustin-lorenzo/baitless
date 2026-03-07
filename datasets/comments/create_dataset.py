import pandas as pd
import json

tlabel2jlabel = {'slippery_slope': 'slippery slope',
                 'population': 'appeal to majority',
                 'authority': 'appeal to authority',
                 'tradition': 'appeal to tradition',
                 'natural': 'appeal to nature',
                 'worse_problems': 'appeal to worse problems',
                 'hasty_generalization': 'hasty generalization',
                 'blackwhite': 'false dilemma',
                 'none': 'none'}
jlabel2tlabel = {'slippery slope': 'slippery_slope',
                 'appeal to majority': 'population',
                 'appeal to authority': 'authority',
                 'appeal to tradition': 'tradition',
                 'appeal to nature': 'natural',
                 'appeal to worse problems': 'worse_problems',
                 'hasty generalization': 'hasty_generalization',
                 'false dilemma': 'blackwhite',
                 'none': 'none'}

all_data = []

# Read tsv data from sahai-etal-2021
txt_df = pd.read_csv('all.txt', sep='\t', on_bad_lines='skip')
for index, row in txt_df.iterrows():
    # Locate tokens and their labels
    words = eval(row.iloc[1])
    labels = eval(row.iloc[4])

    # Identify variables that don't need spaces
    punc = "\"\'.,:?;"
    contraction = "n\'t"
    
    comment = ""    
    current_label = 'none'
    for i in range(len(words)):
        if labels[i] == 'none': # Only add tokens that correspond to fallacies
            continue

        # Avoid space if unnecessary
        current_label = labels[i]
        if (words[i] != contraction and
            words[i][0] not in punc
            and i > 0 and words[i-1] != ('\'' or "\"")):
            comment += " "
        comment += words[i]

    if comment:
        print(comment + "\n" + current_label.upper() + "\n")
        all_data.append([comment, tlabel2jlabel[current_label]])

        
# Read json data from yeh2024
json_array = []
file_names = ['train.json', 'test.json', 'dev.json']
for json_file in file_names:
    with open(json_file, 'r') as f:
        for j in json.load(f):
            json_array.append(j)

for j in json_array:
    entries = j['comments']
    for e in entries:
        comment = e['comment']
        fallacy = e['fallacy']
        if fallacy == 'none':
            continue # model should assume that text has a fallacy (for now)
        all_data.append([comment, fallacy])

# Print to confirm
print(f'{len(all_data)=}')
all_df = pd.DataFrame(all_data, columns=["text", "labels"])
print("\nHEAD:\n=====")
print(all_df.head(2))
print("\nTAIL:\n=====")
print(all_df.tail(2))
print(f"\n\nUNIQUE LABELS:\n{all_df['labels'].unique()}")

# Save as .csv file
all_df.to_csv('all_comments.csv')

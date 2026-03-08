import pandas as pd
import json
from rich import print

plabel2label = {'Black-and-White_Fallacy': 'false dilemma',
                'Whataboutism': 'appeal to worse problems',
                'Bandwagon': 'appeal to majority',
                'Causal_Oversimplification': 'slippery slope',
                'Appeal_to_Authority': 'appeal to authority',
                'Name_Calling': 'ad hominem'}
tlabel2label = {'slippery_slope': 'slippery slope',
                'population': 'appeal to majority',
                'authority': 'appeal to authority',
                'tradition': 'appeal to tradition',
                'natural': 'appeal to nature',
                'worse_problems': 'appeal to worse problems',
                'hasty_generalization': 'hasty generalization',
                'blackwhite': 'false dilemma',
                'none': 'none'}
jlabel2label = {'slippery slope': 'slippery_slope',
                'appeal to majority': 'population',
                'appeal to authority': 'authority',
                'appeal to tradition': 'tradition',
                'appeal to nature': 'natural',
                'appeal to worse problems': 'worse_problems',
                'hasty generalization': 'hasty_generalization',
                'false dilemma': 'blackwhite',
                'none': 'none'}

all_data = []

# Gather instances from propaganda dataset
prop_df = pd.read_csv("propaganda/train-task1-SI.labels", sep='\t', on_bad_lines="skip")

# Keep track of current article and segments without propaganda
article, prev_num, = None, None
non_prop_str, np_curr = "", 0

for index, row in prop_df.iterrows():
    article_num, start, end = row.iloc[0], int(row.iloc[1]), int(row.iloc[2])

    # Change article when article file number changes, add non-propaganda segments
    if index == 0 or article_num != prev_num:
        with open(f"propaganda/train-articles/article{article_num}.txt", 'r') as f:
            article = f.read()
        if non_prop_str:
            np_sentences = non_prop_str.split('. ')
            for s in np_sentences:
                if s and s != '\n':
                    all_data.append([s.strip(), 0])
                    print(f"{s.strip()}")
        non_prop_str = ""
        prev_num = article_num

    # Add propaganda segment from current line
    non_prop_str += article[np_curr:start]
    prop_text = article[start:end]
    np_curr = end
    all_data.append([prop_text, 1])
    print(f"[red]{prop_text}")
    

# Gather instances from quiz dataset
txt_df = pd.read_csv("comments/all.txt", sep='\t', on_bad_lines="skip")
for index, row in txt_df.iterrows():
    # Locate tokens and their labels
    words = eval(row.iloc[1])
    labels = eval(row.iloc[4])

    # Identify variables that don't need spaces
    punc = "\"\'.,:?;"
    contraction = "n\'t"
    
    comment = ""    
    prev_label = ""

    for i in range(len(words)):
        # Add previous segment of comment when label changes
        if i > 0 and labels[i] != prev_label:
            if comment:
                binary = 0 if prev_label == "none" else 1
                all_data.append([comment, binary])
                if binary:
                    print(f"[red]{comment}")
                else:
                    print(f"{comment}")
            comment = ""

        # Add space character when needed
        if (words[i] != contraction and
            words[i][0] not in punc
            and i > 0 and words[i-1] != ('\'' or "\"")):
            comment += " "
        comment += words[i]
        prev_label = labels[i]

    # Add last comment
    if comment:
        binary = 0 if prev_label == "none" else 1
        all_data.append([comment, binary])

        
# Gather instances from comment data
json_array = []
file_names = ['comments/train.json', 'comments/test.json', 'comments/dev.json']
for json_file in file_names:
    with open(json_file, 'r') as f:
        for j in json.load(f):
            json_array.append(j)

for j in json_array:
    entries = j['comments']
    for e in entries:
        comment = e['comment']
        fallacy = e['fallacy']
        binary = 0 if fallacy == "none" else 1
        color_str = "[red]" if binary else "[green]"
        print(f"{comment} {color_str}{fallacy=}")
        all_data.append([comment, binary])
        

# Create dataframe
all_df = pd.DataFrame(all_data, columns=["text", "labels"])

# Print to confirm
print(f'{len(all_data)=}')
print("\nhead:")
print(all_df.head(2))
print("\ntail:")
print(all_df.tail(2))
print(f"\n\n{all_df['labels'].explode().value_counts()}")
print(f"==> total: {len(all_data)}")

# Save as .csv file
path = 'fallacy_binaries.csv'
all_df.to_csv(path)
print(f"dataset saved to {path}.")

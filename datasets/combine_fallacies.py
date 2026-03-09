import pandas as pd
import json
from rich import print

plabel2label = {'Name_Calling,Labeling': 'ad hominem',
                'Whataboutism,Straw_Men,Red_Herring': 'appeal to worse problems',
                'Bandwagon,Reductio_ad_Hitlerum': 'appeal to majority',
                'Exaggeration,Minimisation': 'exaggeration',  # British spelling
                'Black-and-White_Fallacy': 'false dilemma',
                'Causal_Oversimplification': 'slippery slope',
                'Appeal_to_Authority': 'appeal to authority',
                'Flag-Waving': 'flag waving',
                'Appeal_to_fear-prejudice': 'appeal to fear prejudice',
                'Thought-terminating_Cliches': 'thought terminating cliches',
                'Reductio_ad_hitlerum': 'appeal to majority',
                'Bandwagon': 'appeal to majority',
                'Loaded_Language': 'loaded language',
                'Repetition': 'repetition',
                'Doubt': 'doubt',
                'Slogans': 'slogans',
                'Straw_Men': 'appeal to worse problems',
                'Red_Herring': 'appeal to worse problems',
                'Whataboutism': 'appeal to worse problems',
                }
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

# Gather instances from propaganda dataset
prop_df = pd.read_csv('propaganda/train-task2-TC.labels', sep='\t', on_bad_lines='skip')
article, prev_num = None, None
for index, row in prop_df.iterrows():
    article_num, labels, start, end = row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]

    if index == 0 or article_num != prev_num:
        with open(f"propaganda/train-articles/article{article_num}.txt", 'r') as f:
            article = f.read()
            prev_num = article_num

    text = article[start:end]
    if labels in plabel2label:
        labels = [plabel2label[labels]]
    else:
        labels = [plabel2label.get(l, l.lower().replace('-', ' ').replace('_', ' ')) for l in labels.split(',')]

    print(f"[bold yellow]{text=},  [red]{labels=}")
    #print(f"{text=},  {labels=}")
    all_data.append([text, labels])

len_prop = len(all_data)

# Gather instances from quiz dataset
txt_df = pd.read_csv('comments/all.txt', sep='\t', on_bad_lines='skip')
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
        print(f"[bold yellow]{comment} \n [red]{current_label.upper()} \n")
        all_data.append([comment, [tlabel2jlabel[current_label]]])

len_quiz = len(all_data) - len_prop

# Gather instances from comment dataset
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
        if fallacy == 'none':
            continue # model should assume that text has a fallacy (for now)
        print(f"[bold yellow]{comment=} [red]{fallacy=}")
        all_data.append([comment, [fallacy]])

len_comment = len(all_data) - len_prop - len_quiz

# Create dataframe
all_df = pd.DataFrame(all_data, columns=["text", "labels"])

# Print to confirm
print(f'{len(all_data)=}')
print("\nhead:")
print(all_df.head(2))
print("\ntail:")
print(all_df.tail(2))
print(f"\n\n{all_df['labels'].explode().value_counts()}")
print(f"\n\npropaganda instances: {len_prop} \t+ quiz instances: {len_quiz} \t+ comment instances: {len_comment}")
print(f"==> total: {len(all_data)}")

# Save as .csv file
path = 'all_fallacies.csv'
all_df.to_csv(path)
print(f"dataset saved to {path}.")

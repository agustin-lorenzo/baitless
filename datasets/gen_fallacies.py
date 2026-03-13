import random as r
import pandas as pd
from rich import print
from ollama import chat
from ollama import ChatResponse
from concurrent.futures import ThreadPoolExecutor, as_completed


fallacies = [
    'loaded language',
    'slippery slope',
    'ad hominem',
    'appeal to worse problems',
    'appeal to majority',
    'false dilemma',
    'appeal to authority',
    'appeal to nature',
    'appeal to tradition',
    'hasty generalization',
    'repetition',
    'doubt',
    'exaggeration',
    'appeal to fear prejudice',
    'flag waving',
    'slogans',
    'thought terminating cliches'
]

prompt = """
I need data to train a binary classifier to detect whether a piece of text has or doesn't have a fallacy within it. I have non-fallacy instances, but I need examples with fallacies in them. So, I'm going to give you text without a fallacy present (it may or may not be a complete sentence), and you must give me back the text with a specified fallacy inserted into it, while keeping the text and writing style as similar as possible.

I want you to just IMMEDIATELY provide your answer, without any statements beforehand, and I want you to mark the end of your statmentwith this symbol: '-=-'.

The sentence that you have to add fallacies to is as follows:
"""

def generate_fallacies(args):
    i, row = args
    labels = r.sample(fallacies, r.randint(1, 3))
    full_prompt = prompt + row['text']
    full_prompt += f"\nThe fallacy(s) you must add are: {labels}"

    response: ChatResponse = chat(model='gemma3:4b', messages=[
        {'role': 'user', 'content': full_prompt}
    ])
    return i, response.message.content, labels

df = pd.read_csv('fallacy_binaries.csv')
all_data = []

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(generate_fallacies, (i, row)): i for i, row in df.iterrows()}
    for future in as_completed(futures):
        i, content, labels = future.result()
        cleaned = content.replace('-=-', '').strip()
        all_data.append({'text': cleaned, 'labels': labels})
        print(f"\nRow {i}\n========: \noriginal: {df.loc[i]['text']} \nfallacies:[red]{labels}[/red] \nresponse: [bold yellow]{cleaned}[/bold yellow]")

output_df = pd.DataFrame(all_data, columns=['text', 'labels'])
output_df.to_csv('gen_fallacies.csv', index=False)
print("Saved to gen_fallacies.csv.")

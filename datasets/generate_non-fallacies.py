from ollama import chat
from ollama import ChatResponse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt = """
I need data to train a binary classifier to detect whether a piece of text has or doesn't have a fallacy within it. I have the fallacies, but I need the non-fallacy instances. So, I'm gonna give you a fallacy, and I want you to give me back a sentence that removes the fallacy.

I wan't you to just IMMEDIATELY provide your answer, without any statements beforehand, and I want you to mark the end of your statement with this symbol: '-=-'.

The sentence with the fallacy to remove is as follows:
"""

# df = pd.read_csv('all_fallacies.csv')
# for i, row in df.iterrows():
#     full_prompt = prompt + row['text']
#     full_prompt += f"\nThe fallacy(s) to remove is: {row['labels']}"
#     full_prompt += "\nPlease provide the sentence without the fallacy now."


#     response: ChatResponse = chat(model='gemma3:4b', messages=[
#         {
#             'role': 'user',
#             'content': full_prompt
#         },
#     ])
#     print(response.message.content)
#     print()

def generate_non_fallacy(args):
    i, row = args
    full_prompt = prompt + row['text']
    full_prompt += f"\nThe fallacy(s) to remove is: {row['labels']}"
    full_prompt += "\nPlease provide the sentence without the fallacy now."
    
    response: ChatResponse = chat(model='gemma3:4b', messages=[
        {'role': 'user', 'content': full_prompt}
    ])
    return i, response.message.content

df = pd.read_csv('all_fallacies.csv')
all_data = []

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(generate_non_fallacy, (i, row)): i for i, row in df.iterrows()}
    for future in as_completed(futures):
        i, content = future.result()
        cleaned = content.replace('-=-', '').strip()
        all_data.append({'text': cleaned, 'labels': 0})
        print(f"Row {i}: {content}")

output_df = pd.DataFrame(all_data, columns=['text', 'labels'])
output_df.to_csv('non_fallacies.csv', index=False)
print("Saved to non_fallacies.csv")

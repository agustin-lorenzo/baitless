file_path = "datasets/train-articles/article111111111.txt"

with open(file_path, 'r') as f:
    content = f.read()

print(content[2023:2086])

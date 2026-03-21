import pandas as pd

gf = pd.read_csv('gen_fallacies.csv')
gf['labels'] = 1
print(f"{gf=}")
gnf = pd.read_csv('gen_non-fallacies.csv')

rb = pd.read_csv('real_binaries.csv')

all_gen = pd.concat([gf, gnf, rb], ignore_index=True)
all_gen = all_gen.iloc[:, :-1]
print(all_gen)



all_gen.to_csv('all_binaries.csv')

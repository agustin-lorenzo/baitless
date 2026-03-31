from rich import print
from rich.console import Console
from rich.traceback import install
from transformers import pipeline
install()

binary = pipeline('text-classification', model='models/f-binary_db-model')
classes = pipeline('text-classification', model='models/f-class_model', top_k=3)

while True:
    inputs = input("\n\nInput text to check for fallacies: ")
    console = Console()
    console.rule()
    
    b_result = binary(inputs)
    c_result = classes(inputs)
    
    detected = "[red]FALLACY DETECTED" if b_result[0]['label'] == 'LABEL_1' else "[green]NO FALLACY DETECTED"
    print(f"{detected}")
    print(f"Most likely fallacies:")
    for f in c_result[0]:
        prefix = "[bold magenta]"if f['score'] >= 0.5 else "[grey]"
        print(f"\t{prefix}{f['label']} [white](score: [cyan]{f['score']:.2f}[/cyan])")


import os
import json
import argparse
import timeit
import tqdm
import evaluate
import PIL.Image
from competition_transcriber import CompetitionTranscriber


parser = argparse.ArgumentParser(
    description='Evaluate a competition transcriber OCR.',
)
parser.add_argument(
    'split',
    choices=['dev', 'test'],
    default='dev',
    help='The data split to evaluate on.',
)
args = parser.parse_args()
split = args.split

print('evaluating on', split)
print()

print('loading transcriber')
transcriber = CompetitionTranscriber()

print('loading data set')
with open(os.path.join(split, 'texts.json'), encoding='utf-8') as f:
    data = json.load(f)

print('loading evaluator')
cer_evaluator = evaluate.load('cer')

print('started evaluation')
predictions = []
start_time = timeit.default_timer()
for doc in tqdm.tqdm(data):
    image = PIL.Image.open(os.path.join(split, doc['image']))
    text = transcriber.transcribe(image)
    predictions.append(text)
duration = timeit.default_timer() - start_time
cer = cer_evaluator.compute(
    predictions=predictions,
    references=[doc['text'] for doc in data],
)

print('results:')
print(' CER:', cer)
print(' Duration (sec):', duration)
with open('results.txt', 'a', encoding='utf-8') as f:
    print(cer, duration, sep='\t', file=f)

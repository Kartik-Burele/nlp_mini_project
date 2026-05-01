import json

d = json.load(open(r'k:\MTECH\NLP\Mini Project\SemiSage3.0\SemiSagev3.0\output2\final_dataset.json','r',encoding='utf-8'))

print('=== SHORT ANSWERS (< 10 chars) - spot check ===')
count = 0
for doc in d['data']:
    for p in doc['paragraphs']:
        for qa in p['qas']:
            a = qa['answers'][0]['text']
            if len(a) < 10:
                ctx = p['context']
                start = qa['answers'][0]['answer_start']
                actual = ctx[start:start+len(a)]
                match = 'OK' if actual == a else 'MISMATCH'
                print(f'  [{match}] [{doc["title"]}] Q: {qa["question"][:70]}')
                print(f'    A: "{a}" (len={len(a)}, start={start})')
                count += 1
print(f'\nTotal short answers: {count}')

print('\n=== LONGEST ANSWER ===')
longest = None
for doc in d['data']:
    for p in doc['paragraphs']:
        for qa in p['qas']:
            a = qa['answers'][0]['text']
            if longest is None or len(a) > len(longest[0]):
                longest = (a, qa['question'], doc['title'])
print(f'  [{longest[2]}] Q: {longest[1][:80]}')
print(f'  A ({len(longest[0])} chars): {longest[0][:200]}...')

print('\n=== CONTEXT LENGTH STATS ===')
ctx_lens = []
for doc in d['data']:
    for p in doc['paragraphs']:
        ctx_lens.append(len(p['context']))
print(f'  Min context: {min(ctx_lens)} chars')
print(f'  Max context: {max(ctx_lens)} chars')
print(f'  Avg context: {round(sum(ctx_lens)/len(ctx_lens))} chars')

print('\n=== MANUAL SPOT CHECK: 3 random QAs with answer_start verification ===')
import random
random.seed(42)
all_qas = []
for doc in d['data']:
    for p in doc['paragraphs']:
        for qa in p['qas']:
            all_qas.append((qa, p['context'], doc['title']))

samples = random.sample(all_qas, min(5, len(all_qas)))
for qa, ctx, title in samples:
    a = qa['answers'][0]
    start = a['answer_start']
    text = a['text']
    extracted = ctx[start:start+len(text)]
    match = 'PASS' if extracted == text else 'FAIL'
    print(f'\n  [{match}] [{title}]')
    print(f'  Q: {qa["question"]}')
    print(f'  A: "{text[:100]}{"..." if len(text)>100 else ""}"')
    print(f'  Verify ctx[{start}:{start+len(text)}] = "{extracted[:100]}{"..." if len(extracted)>100 else ""}"')

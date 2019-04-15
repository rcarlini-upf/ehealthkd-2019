import argparse
from collections import OrderedDict
from pathlib import Path

from utils import Collection

CORRECT_A = 'correct_A'
INCORRECT_A = 'incorrect_A'
PARTIAL_A = 'partial_A'
SPURIOUS_A = 'spurious_A'
MISSING_A = 'missing_A'

CORRECT_B = 'correct_B'
SPURIOUS_B = 'spurious_B'
MISSING_B = 'missing_B'

def report(data, verbose):
    for key, value in data.items():
        print(f'{key}: {len(value)}')

    if verbose:
        for key, value in data.items():
            print(f'\n==================={key.upper().center(14)}===================\n')
            if isinstance(value, dict):
                print('\n'.join(f'{x} --> {y}' for x,y in value.items()))
            else:
                print('\n'.join(str(x) for x in value))

def subtaskA(gold, submit, verbose):
    return match_keyphrases(gold, submit)

def match_keyphrases(gold, submit):
    correct = {}
    incorrect = {}
    partial = {}
    spurious = []
    missing = []

    for gold_sent, submit_sent in zip(gold.sentences, submit.sentences):
        if gold_sent.text != submit_sent.text:
            print('[ERROR]: Wrong sentence!')
            continue

        gold_sent = gold_sent.clone(shallow=True)
        submit_sent = submit_sent.clone(shallow=True)

        # correct
        for keyphrase in submit_sent.keyphrases[:]:
            match = gold_sent.find_keyphrase(spans=keyphrase.spans)
            if match and match.label == keyphrase.label:
                correct[keyphrase] = match
                gold_sent.keyphrases.remove(match)
                submit_sent.keyphrases.remove(keyphrase)

        # incorrect
        for keyphrase in submit_sent.keyphrases[:]:
            match = gold_sent.find_keyphrase(spans=keyphrase.spans)
            if match:
                assert match.label != keyphrase.label
                incorrect[keyphrase] = match
                gold_sent.keyphrases.remove(match)
                submit_sent.keyphrases.remove(keyphrase)

        # partial
        for keyphrase in submit_sent.keyphrases[:]:
            match = find_partial_match(keyphrase, gold_sent.keyphrases)
            if match:
                partial[keyphrase] = match
                gold_sent.keyphrases.remove(match)
                submit_sent.keyphrases.remove(keyphrase)

        # spurious
        spurious.extend(submit_sent.keyphrases)

        # missing
        missing.extend(gold_sent.keyphrases)

    return {
        CORRECT_A : correct,
        INCORRECT_A : incorrect,
        PARTIAL_A : partial,
        SPURIOUS_A : spurious,
        MISSING_A : missing,
    }

def find_partial_match(keyphrase, sentence):
    return next((match for match in sentence if partial_match(keyphrase, match)), None)

def partial_match(keyphrase1, keyphrase2):
    match = False
    match |= any(start <= x < end for start, end in keyphrase1.spans for x,_ in keyphrase2.spans)
    match |= any(start <= x < end for start, end in keyphrase2.spans for x,_ in keyphrase1.spans)
    return match

def subtaskB(gold, submit, data, verbose):
    return match_relations(gold, submit, data)

def match_relations(gold, submit, data):
    correct = {}
    spurious = []
    missing = []

    for gold_sent, submit_sent in zip(gold.sentences, submit.sentences):
        if gold_sent.text != submit_sent.text:
            print('[ERROR]: Wrong sentence!')
            continue

        gold_sent = gold_sent.clone(shallow=True)
        submit_sent = submit_sent.clone(shallow=True)

        # correct
        for relation in submit_sent.relations[:]:
            origin = relation.from_phrase
            origin = map_keyphrase(origin, data)

            destination = relation.to_phrase
            destination = map_keyphrase(destination, data)

            if origin is None or destination is None:
                continue

            match = gold_sent.find_relation(origin.id, destination.id, relation.label)
            if match:
                correct[relation] = match
                gold_sent.relations.remove(match)
                submit_sent.relations.remove(relation)

        # spurious
        spurious.extend(submit_sent.relations)

        # missing
        missing.extend(gold_sent.relations)

    return {
        CORRECT_B : correct,
        SPURIOUS_B : spurious,
        MISSING_B : missing,
    }

def map_keyphrase(keyphrase, data):
    try:
        return data[CORRECT_A][keyphrase]
    except KeyError:
        pass
    try:
        return data[PARTIAL_A][keyphrase]
    except KeyError:
        pass
    return None

def main(gold_input, submit_input, skip_A, skip_B, verbose):
    gold = Collection()
    gold.load(gold_input)

    submit = Collection()
    submit.load(submit_input)

    data = OrderedDict()

    dataA = subtaskA(gold, submit, verbose)
    data.update(dataA)
    if not skip_A:
        report(dataA, verbose)

    if not skip_B:
        dataB = subtaskB(gold, submit, dataA, verbose)
        data.update(dataB)
        print()
        report(dataB, verbose)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gold')
    parser.add_argument('submit')
    parser.add_argument('--skip-A', action='store_true')
    parser.add_argument('--skip-B', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(Path(args.gold), Path(args.submit), args.skip_A, args.skip_B, args.verbose)
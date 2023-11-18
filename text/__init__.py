import string

from text import cmudict
from text.cleaners import english_cleaners

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness:
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_arpabet)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


cmu_dict = {}

with open('text/en_dictionary') as f:
    for entry in f:
        tokens = []
        for t in entry.split():
            tokens.append(t)
        cmu_dict[tokens[0]] = tokens[1:]


def text_to_sequence(text, stop_on_word_error=True):
    phoneme = []
    sequence = []
    text = _clean_text(text, ["english_cleaners"])
    text = text.upper()
    text = text.split(' ')

    for phn in text:
        found = False
        if phn.startswith("{"):
            phn = phn.strip().replace('{', '').replace('}', '') + ' '
            phoneme.append(phn)
            continue
        for word, pronunciation in cmu_dict.items():
            if word == phn:
                found = True
                arpa = ''.join(pronunciation) + ' '
                phoneme.append(arpa)
                break

        if not found:
            if phn not in string.punctuation:
                if stop_on_word_error:
                    raise Exception(f'"{phn}" NOT FOUND IN DICTIONARY!')
                print(f'THE WORD "{phn}" WILL BE USED WITHOUT ARPABET PHONEME.')
                phn = str(phn).replace(' ', '')
                phoneme.append(phn + ' ')
            else:
                phoneme.append(phn)

    text = (''.join(phoneme)
            .replace(' ,', ', ')
            .replace(' .', '. ')
            .replace(' !', '!')
            .replace(' ?', '? ')
            .replace(' ;', '; ')
            .replace(' :', ': ')
            .replace(' -', ' - ')
            .strip())

    sequence += _symbols_to_sequence(text)

    # print(sequence_to_text(sequence))

    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'

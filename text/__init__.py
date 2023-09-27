import re
from text.cleaners import english_cleaners
import torch
import numpy as np


_punctuation = ';:,.!?¡¿—"«» '
_letters = 'abcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = list(_letters) + list(_letters_ipa) + list(_punctuation)


print(symbols)

def text_to_sequence_1(text):
    text = text.strip()
    text = english_cleaners(text)
    text = text.replace(' ', '[SPACE]')
    output = tokenizer.encode(text)
    print(output.tokens)
    print(output.ids)
    for tok in output.tokens:
        if tok == '[UNK]':
            raise Exception(f'Unknown symbol {tok} found')
    
    sequence = np.array([output.ids])

    if torch.cuda.is_available():
        return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    else:
        return torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()


def text_to_sequence_2(text):
    text = text.strip()
    text = english_cleaners(text)
    print(text)
    
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    for s in text:
        if s not in symbol_to_id:
            print(f'{s} not found!')
            raise Exception(f'{s} not found!')
    
    sequence = np.array([[symbol_to_id[s] for s in text if s in symbol_to_id]])
    print(sequence)
    if torch.cuda.is_available():
        return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    else:
        return torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()


def sequence_to_text(tokens):
    txt = ''.join(tokens)
    txt = txt.replace('[SPACE]', ' ')
    print(txt)


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
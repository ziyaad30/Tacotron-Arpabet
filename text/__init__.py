import re
from text.cleaners import english_cleaners
import torch
import numpy as np


_punctuation = ';:,.!?¡¿—"«» '
_letters = 'abcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = list(_letters) + list(_letters_ipa) + list(_punctuation)


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

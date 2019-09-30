import re

import numpy as np
import tensorflow as tf



# get wordshape of a specific word
def get_wordshape(word):
    wordshape = []
    for char in word:
        if re.match('[0-9]', char):
            wordshape.append(3)
        elif re.match('[A-Z]', char):
            wordshape.append(1)
        elif re.match('[a-z]', char):
            wordshape.append(2)
        else:
            wordshape.append(4)
    return wordshape


def valid_numeric_list(value):
    try:
        value = list(map(float, value))
        total = sum(value)
        return [v / total for v in value]

    except ValueError as e:
        raise ValueError("list of numeric values expected : got {}".format(e.args[0]))


def iob_mask(vocab, start, end, pad=None):
    small = 0
    mask = np.ones((len(vocab), len(vocab)), dtype=np.float32)
    for from_ in vocab:
        for to in vocab:
            # Can't move to start
            if to is start:
                mask[vocab[to], vocab[from_]] = small
            # Can't move from end
            if from_ is end:
                mask[vocab[to], vocab[from_]] = small
            # Can only move from pad to pad or end
            if from_ is pad:
                if not (to is pad or to is end):
                    mask[vocab[to], vocab[from_]] = small
            elif from_ is start:
                # Can't move from start to a B
                if to.startswith("B-"):
                    mask[vocab[to], vocab[from_]] = small
            else:
                if from_.startswith("B-"):
                    # Can't move from a B to a B of another type
                    if to.startswith("B-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("I-"):
                    # Can't move from an I to a B of another type
                    if to.startswith("B-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("O"):
                    # Can't move from an O to a B
                    if to.startswith("B-"):
                        mask[vocab[to], vocab[from_]] = small
    return mask


def iob2_mask(vocab, start, end, pad=None):
    small = 0
    mask = np.ones((len(vocab), len(vocab)), dtype=np.float32)
    for from_ in vocab:
        for to in vocab:
            # Can't move to start
            if to is start:
                mask[vocab[to], vocab[from_]] = small
            # Can't move from end
            if from_ is end:
                mask[vocab[to], vocab[from_]] = small
            # Can only move from pad to pad or end
            if from_ is pad:
                if not (to is pad or to is end):
                    mask[vocab[to], vocab[from_]] = small
            elif from_ is start:
                # Can't move from start to a I
                if to.startswith("I-"):
                    mask[vocab[to], vocab[from_]] = small
            else:
                if from_.startswith("B-"):
                    # Can't move from a B to an I of a different type
                    if to.startswith("I-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("I-"):
                    # Can't move from an I to an I of another type
                    if to.startswith("I-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("O"):
                    # Can't move from an O to an I
                    if to.startswith("I-"):
                        mask[vocab[to], vocab[from_]] = small
    return (mask)


def iobes_mask(vocab, start, end, pad=None):
    small = 0
    mask = np.ones((len(vocab), len(vocab)), dtype=np.float32)
    for from_ in vocab:
        for to in vocab:
            # Can't move to start
            if to is start:
                mask[vocab[to], vocab[from_]] = small
            # Can't move from end
            if from_ is end:
                mask[vocab[to], vocab[from_]] = small
            # Can only move from pad to pad or to end
            if from_ is pad:
                if not (to is pad or to is end):
                    mask[vocab[to], vocab[from_]] = small
            elif from_ is start:
                # Can't move from start to I or E
                if to.startswith("I-") or to.startswith("E-"):
                    mask[vocab[to], vocab[from_]] = small
            else:
                if from_.startswith("B-"):
                    # Can't move from B to B, S, O, End, or Pad
                    if (
                            to.startswith("B-") or
                            to.startswith("S-") or
                            to.startswith("O") or
                            to is end or
                            to is pad
                    ):
                        mask[vocab[to], vocab[from_]] = small
                    # Can only move to matching I or E
                    elif to.startswith("I-") or to.startswith("E-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("I-"):
                    # Can't move from I to B, S, O, End or Pad
                    if (
                            to.startswith("B-") or
                            to.startswith("S-") or
                            to.startswith("O") or
                            to is end or
                            to is pad
                    ):
                        mask[vocab[to], vocab[from_]] = small
                    # Can only move to matching I or E
                    elif to.startswith("I-") or to.startswith("E-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif (
                        from_.startswith("E-") or
                        from_.startswith("I-") or
                        from_.startswith("S-") or
                        from_.startswith("O")
                ):
                    # Can't move from E to I or E
                    # Can't move from I to I or E
                    # Can't move from S to I or E
                    # Can't move from O to I or E
                    if to.startswith("I-") or to.startswith("E-"):
                        mask[vocab[to], vocab[from_]] = small
    return mask


def transition_mask_np(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a CRF mask.
    Returns a mask with invalid moves as 0 and valid as 1.
    :param vocab: dict, Label vocabulary mapping name to index.
    :param span_type: str, The sequence labeling formalism {IOB, IOB2, BIO, or IOBES}
    :param s_idx: int, What is the index of the GO symbol?
    :param e_idx: int, What is the index of the EOS symbol?
    :param pad_idx: int, What is the index of the PAD symbol?
    Note:
        In this mask the PAD symbol is between the last symbol and EOS, PADS can
        only move to pad and the EOS. Any symbol that can move to an EOS can also
        move to a pad.
    """
    rev_lut = {v: k for k, v in vocab.items()}
    start = rev_lut[s_idx]
    end = rev_lut[e_idx]
    pad = None if pad_idx is None else rev_lut[pad_idx]
    if span_type.upper() == "IOB":
        mask = iob_mask(vocab, start, end, pad)
    if span_type.upper() == "IOB2" or span_type.upper() == "BIO":
        mask = iob2_mask(vocab, start, end, pad)
    if span_type.upper() == "IOBES":
        mask = iobes_mask(vocab, start, end, pad)
    return mask


def transition_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a CRF Mask.
    Returns a mask with invalid moves as 0 and valid moves as 1.
    """
    mask = transition_mask_np(vocab, span_type, s_idx, e_idx, pad_idx).T
    inv_mask = (mask == 0).astype(np.float32)
    return tf.constant(mask), tf.constant(inv_mask)


if __name__ == '__main__':
    S = '[CLS]'
    E = '[SEP]'
    P = '<PAD>'
    SPAN_TYPE = "IOB2"
    label_vocab = {'<PAD>': 0, 'X': 1, '[CLS]': 2, '[SEP]': 3, 'B-PosReg': 4, 'I-Interaction': 5, 'I-Pathway': 6,
                   'B-Var': 7, 'B-NegReg': 8, 'I-Var': 9, 'B-Enzyme': 10, 'I-Reg': 11, 'I-PosReg': 12, 'B-Pathway': 13,
                   'I-Enzyme': 14, 'I-MPA': 15, 'B-Disease': 16, 'B-Reg': 17, 'I-Gene': 18, 'I-Disease': 19,
                   'B-MPA': 20, 'I-Protein': 21, 'B-Interaction': 22, 'B-CPA': 23, 'B-Gene': 24, 'O': 25,
                   'B-Protein': 26, 'I-CPA': 27, 'I-NegReg': 28}
    # label_vocab = {
    #     '<PAD>': 0, "B": 1, "I": 2, "O": 3, "X": 4, "[CLS]": 5, "[SEP]": 6
    # }
    mask = transition_mask_np(label_vocab, SPAN_TYPE, label_vocab[S], label_vocab[E], label_vocab[P])
    print(mask[:, label_vocab['X']])

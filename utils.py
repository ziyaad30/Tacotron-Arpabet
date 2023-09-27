from tacotron2_model.utils import get_mask_from_lengths

LEARNING_RATE_PER_64 = 4e-4
MAXIMUM_LEARNING_RATE = 4e-4


def load_labels_file(filepath):
    """
    Load labels file

    Parameters
    ----------
    filepath : str
        Path to text file

    Returns
    -------
    list
        List of samples
    """
    with open(filepath, encoding='utf-8') as f:
        return [line.strip().split("|") for line in f]



def train_test_split(filepaths_and_text, train_size):
    """
    Split dataset into train & test data

    Parameters
    ----------
    filepaths_and_text : list
        List of samples
    train_size : float
        Percentage of entries to use for training (rest used for testing)

    Returns
    -------
    (list, list)
        List of train and test samples
    """
    train_cutoff = int(len(filepaths_and_text) * train_size)
    train_files = filepaths_and_text[:train_cutoff]
    test_files = filepaths_and_text[train_cutoff:]
    print(f"{len(train_files)} train files, {len(test_files)} test files")
    return train_files, test_files



def get_learning_rate(batch_size):
    """
    Calulate learning rate.

    Parameters
    ----------
    batch_size : int
        Batch size

    Returns
    -------
    float
        Learning rate
    """
    return min(
        (batch_size / 64) ** 0.5 * LEARNING_RATE_PER_64,  # Adam Learning Rate is proportional to sqrt(batch_size)
        MAXIMUM_LEARNING_RATE,
    )
    
    
def calc_avgmax_attention(mel_lengths, text_lengths, alignment):
    """
    Calculate Average Max Attention for Tacotron2 Alignment.
    Roughly represents how well the model is linking the text to the audio.
    Low values during training typically result in unstable speech during inference.

    Parameters
    ----------
    mel_lengths : torch.Tensor
        lengths of each mel in the batch
    text_lengths : torch.Tensor
        lengths of each text in the batch
    alignment : torch.Tensor
        alignments from model of shape [B, mel_length, text_length]

    Returns
    -------
    float
        average max attention
    """
    mel_mask = get_mask_from_lengths(mel_lengths, device=alignment.device)
    txt_mask = get_mask_from_lengths(text_lengths, device=alignment.device)
    # [B, mel_T, 1] * [B, 1, txt_T] -> [B, mel_T, txt_T]
    attention_mask = txt_mask.unsqueeze(1) & mel_mask.unsqueeze(2)

    alignment = alignment.data.masked_fill(~attention_mask, 0.0)
    # [B, mel_T, txt_T]
    avg_prob = alignment.data.amax(dim=2).sum(1).div(mel_lengths.to(alignment)).mean().item()
    return avg_prob
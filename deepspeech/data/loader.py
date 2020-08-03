import torch
from torch.nn.utils.rnn import pad_sequence


def collate_input_sequences(samples):
    """Returns a batch of data given a list of samples.

    Args:
        samples: List of (x, y) where:

             x : A tuple:
                -  torch.Tensor : an input sequence to the network with size
                       (len(torch.Tensor), n_features) .
                -  int : the length of the corresponding output sequence
                      produced by the network given the  torch.Tensor  as
                      input.
             y : A  torch.Tensor  containing the target output sequence.

    Returns:
        A tuple of  ((batch_x, batch_out_lens), batch_y)  where:

            batch_x: The concatenation of all  torch.Tensor "s in  x  along a
                new dim in descending order by  torch.Tensor  length.

                This results in a  torch.Tensor  of size (L, N, D) where L is
                the maximum  torch.Tensor  length, N is the number of samples,
                and D is n_features.

                 torch.Tensor "s shorter than L are extended by zero padding.

            batch_out_lens: A  torch.IntTensor  containing the  int  values
                from  x  in an order that corresponds to the samples in
                 batch_x .

            batch_y: A list of  torch.Tensor  containing the  y   torch.Tensor 
                sequences in an order that corresponds to the samples in
                 batch_x .

    
    """

    samples = [(*x, y) for x, y in samples]
    sorted_samples = sorted(samples, key=lambda s: len(s[0]), reverse=True)

    seqs, seq_lens, labels = zip(*sorted_samples)

    x = (pad_sequence(seqs), torch.IntTensor(seq_lens))
    y = list(labels)

    return x, y

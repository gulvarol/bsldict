"""Watch-Read-Lookup loss implementation for:
    "Watch, read and lookup: learning to spotsigns from multiple supervisors", ACCV 2020

The accompanying `sample_inputs.pkl` can be used as an input to the loss to illustrate
dimensions.  Note: this function is included to help clarify implementation details.

Tested with PyTorch 1.4.
"""
import argparse
import torch
import torch.nn.functional as F


class WatchReadLookupLoss(torch.nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, bsl1k_max_len: int = 13, device: str = "cuda"):
        """Compute the WatchReadLookup Loss.

        Args:
             x (dict): a datastructure representing N embeddings and their associated
                 meta data, which is organised as follows:
                 - "batch_labels" (Tensor of ints with shape (N,)) mapping each embedding
                     to its batch index.
                 - "domain_labels" (Tensor of doubles with shape (N,)) mapping each
                     embedding to its domain and taking values in {0, 1} where label 0
                     indicates the bsl1k domain and label 1 indicates the dictionary
                     domain.
                 - "is_mouthing" (Tensor of doubles with shape (N,)) indicating whether
                     the embedding corresonds to a mouthing location.
                 - "targets" (Tensor of ints with shape (N,)) indicating the class labels
                     of the embeddings where available (and -1 for bsl1k embeddings if
                     unknown).
                 - "features" (Tensor of floats with shape (N,256)) the embedidngs
                     produced by the network.
            bsl1k_max_len (int): The max number of words that can appear in a bsl1k dict
                (predefining this value allows the loss to be made more efficient).
            device (str): device on which processing should be done
        """
        # True if bsl1k video
        ix_bsl1k = x["domain_labels"] == 0
        # True if dictionary video
        ix_dict = x["domain_labels"] == 1
        # sanity check that only two domains where specified
        assert (x["domain_labels"].unique().cpu() == torch.Tensor([0, 1])).all(), (
            "Expected to have domain labels from two domains, but found "
            f"{x['domain_labels'].unique()}"
        )
        # reshape features -> N x 256
        features = x["features"].squeeze(-1).squeeze(-1).squeeze(-1)
        # L2 normalize each feature vector
        features = F.normalize(features, p=2, dim=1)
        # obtain features corresponding to bsl1k videos (num_bsl1k x 256 if neg_window=0)
        embd_bsl1k = features[ix_bsl1k]
        # obtain features which correspond to sdict videos (num_dict x 256)
        embd_dict = features[ix_dict]
        # obtain list of classes which correspond to bsl1k videos
        targets_bsl1k = x["targets"][ix_bsl1k]
        # obtain list of classes which correspond to sdict videos
        targets_dict = x["targets"][ix_dict]
        # obtain batch indices which correspond to bsl1k videos
        batch_bsl1k = x["batch_labels"][ix_bsl1k]
        # obtain batch indices which correspond to sdict videos
        batch_dict = x["batch_labels"][ix_dict]
        # obtain indicators for which dict videos have mouthing
        is_mouthing_dict = x["is_mouthing"][ix_dict]

        # cosine distance matrix of [num_bsl1k videos x num_dict videos]
        # features are normalised beforehand
        distances = torch.matmul(embd_bsl1k, embd_dict.t())

        # fixed temperature
        distances /= self.temperature

        # === COMPUTE MATCH_MULTI ===
        # Here we compute a matrix indicating which pairs of embeddings are positives
        # and which are negatives for the MIL NCE loss.

        # We first create a matrix to store the dictionary class labels corresponding
        # to each mouthing in the bsl1k sentence (this is stored in the zeroth column)
        # followed by the class labels of all remaining words we expect to find in the
        # sentence (up to a maximum of `bsl1k_max_len`)
        num_bsl1k_embeddings = len(targets_bsl1k)
        targets_bsl1k_multi = -1 * torch.ones(num_bsl1k_embeddings, bsl1k_max_len,
                                              device=device, dtype=torch.int)
        cnt = 0
        for i in batch_bsl1k.unique():
            # select the dictionary entries corresponding to the current batch
            dict_mask = batch_dict == i
            # Find the word that corresponds to the mouthing location
            mouthing_word = targets_dict[dict_mask * (is_mouthing_dict == 1)].unique()
            # Find the words that corresponds to the background
            back_subtitle_words = (targets_dict[dict_mask * (is_mouthing_dict == 0)]
                                   ).unique()
            # Find the class labels for the bsl1k sentence words (from subtitles)
            targets_bsl1k_batch = targets_bsl1k[batch_bsl1k == i]

            # Count the total number of embeddings in the current batch
            T = len(targets_bsl1k_batch)

            # create a mask to identify the locations in target matrix which
            # correspond to the current batch embeddings
            bsl1k_batch_mask = torch.arange(cnt, cnt + T)

            # find the position of the word which corresponds to the mouthing in the
            # targets vector
            bsl1k_mouthing_targets_batch = bsl1k_batch_mask[targets_bsl1k_batch != -1]
            # find the position of the words which corresponds to the background (i.e.
            # not mouthing) in the # targets vector
            bsl1k_background_targets_batch = bsl1k_batch_mask[targets_bsl1k_batch == -1]

            # Assign the dictionary class labels into the target matrix
            targets_bsl1k_multi[bsl1k_mouthing_targets_batch, 0] = mouthing_word.item()
            targets_bsl1k_multi[bsl1k_background_targets_batch, :len(back_subtitle_words)
                                ] = back_subtitle_words.int()
            cnt += T

        # Next we expand the label matrix along an additional dimension to incorporate
        # dictionary labels and create a 3D tensor of booleans where True values
        # correspond to locations where the label assigned to the bsl1k embedding
        # matches the dictionary embedding label
        match_multi = targets_bsl1k_multi.unsqueeze(2) - targets_dict == 0

        # Look for any possible positives along the max_len/num_subtitles dimension.
        # (this changes the shape of match_multi to [bsl1k x Dict])
        match_multi = match_multi.any(1)

        # === MAKE NUMERATOR/DENOMINATOR MASKS ===
        with torch.no_grad():
            # Whether the bsl1k and dict pairs belong to the same batch (a matrix of
            # size num_bsl1k x num_dict)
            batch_match = (batch_bsl1k.unsqueeze(1) - batch_dict == 0)
            # identify embedding pairs that have the same class label, but fall within
            # different batches (size num_bsl1k x num_dict)
            diff_batch_match = match_multi * ~batch_match

            # create masks for the different terms that comprise the MIL NCE loss
            num_dict_embeddings = len(targets_dict)
            pos_mask = torch.zeros(num_bsl1k_embeddings, num_dict_embeddings,
                                   device=device, dtype=bool)

            mask_list = []
            pos_mask_list = []
            num_unique_dicts = targets_dict.unique_consecutive()

            # Loop over the dictionary embeddings, grouped by class label
            for i, t in enumerate(num_unique_dicts):
                # find the set of pairs with the current dictionary class label
                curr_dict = targets_dict == t

                # find the bsl1k embeddings that share the same class label
                curr_bsl1k = match_multi[:, curr_dict][:, 0]

                # positives correspond to locations for the current batch that share
                # the same label
                where_dict_pos = torch.where(curr_dict)[0]
                where_bsl1k_pos = torch.where(curr_bsl1k)[0]
                num_dict_pos = where_dict_pos.numel()
                num_bsl1k_pos = where_bsl1k_pos.numel()
                where_bsl1k_pos = where_bsl1k_pos.repeat(num_dict_pos, 1).t().flatten()
                where_dict_pos = where_dict_pos.repeat(num_bsl1k_pos).flatten()
                pos_mask_list.append((where_bsl1k_pos.cpu(), where_dict_pos.cpu()))

                # wipe the masks ready for the next batch
                pos_mask.fill_(False)

                # Account for matches that occur in different batches
                pos_neg_mask = (curr_dict.unsqueeze(0) | curr_bsl1k.unsqueeze(1))
                pos_neg_mask *= ~diff_batch_match
                where_mask = torch.where(pos_neg_mask)
                mask_list.append((where_mask[0].cpu(), where_mask[1].cpu()))

        numerator = torch.empty(len(num_unique_dicts), device=device)
        denominator = torch.empty(len(num_unique_dicts), device=device)
        for i, t in enumerate(num_unique_dicts):
            numerator[i] = torch.logsumexp(distances[pos_mask_list[i]], dim=0)
            denominator[i] = torch.logsumexp(distances[mask_list[i]], dim=0)

        return torch.mean(denominator - numerator)


if __name__ == "__main__":
    # Sample usage
    import pickle as pkl
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    outputs = pkl.load(open("sample_inputs.pkl", "rb"))
    dev = torch.device(args.device)
    if "is_mouthing" not in outputs:
        outputs["is_mouthing"] = torch.ones(outputs["targets"].size())
    outputs = {k: v.to(dev) for k, v in outputs.items()}
    criterion = WatchReadLookupLoss()
    loss = criterion(x=outputs)
    print(f"Loss for samples: {loss}")

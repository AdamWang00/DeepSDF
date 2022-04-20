import torch

class ModelLatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
    ):
        self.data_source = data_source

        data = torch.load(data_source)
        if isinstance(data["latent_codes"], torch.Tensor):
            num_vecs = data["latent_codes"].size()[0]
            lat_vecs = []
            for i in range(num_vecs):
                lat_vecs.append(data["latent_codes"][i].cuda())
            self.data = lat_vecs
        else:
            num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape
            lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)
            lat_vecs.load_state_dict(data["latent_codes"])
            self.data = lat_vecs.weight.data.detach()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
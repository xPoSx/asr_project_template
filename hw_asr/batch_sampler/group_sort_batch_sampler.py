from torch.utils.data import Sampler
import torch


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batches_per_group=20):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches_per_group = batches_per_group
        self.elem_to_group = torch.zeros(len(data_source))
        self.n_groups = len(data_source) // batch_size //  batches_per_group

        sorted_i = torch.argsort(torch.tensor([len(x['audio']) for x in data_source]))
        for i in range(self.n_groups):
            group_ids = sorted_i[i * batch_size*batches_per_group:(i + 1) * batch_size*batches_per_group]
            self.elem_to_group[group_ids] = i

    def __iter__(self):
        sampled = 0
        inds = torch.arange(len(self.data_source))
        while sampled < len(self.data_source):
            group_id = int(torch.randint(0, self.n_groups, (1,))[0])
            cur_group = inds[self.elem_to_group == group_id]
            batch_inds = torch.randperm(len(cur_group))[:self.batch_size]
            batch = cur_group[batch_inds].tolist()
            sampled += self.batch_size
            yield batch




    def __len__(self):
        return len(self.data_source)

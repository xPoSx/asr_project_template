from torch.utils.data import BatchSampler
import torch


class GroupLengthBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size, batches_per_group=20):
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
        num_yielded = 0
        inds = torch.arange(len(self.data_source))
        while num_yielded < len(self.data_source):
            group_id = torch.randint(0, self.n_groups + 1, (1,))
            cur_group = inds[self.elem_to_group == group_id]
            batch_ids = torch.randperm(len(cur_group))[:self.batch_size]
            batch = cur_group[batch_ids]
            num_yielded += len(batch)
            print(batch.size())
            yield batch

        # for i in range(self.n_groups + 1):
        #     cur_group = inds[self.elem_to_group == i]
        #     # print(len(cur_group))
        #     for k in range(len(cur_group) // self.batch_size):
        #         batch = cur_group[k * self.batch_size:(k + 1) * self.batch_size].tolist()
        #         yield batch

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

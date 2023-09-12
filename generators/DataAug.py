import torch

tensor = torch.arange(0, 15).reshape(3, -1)
print(tensor)

row_list = []
for index, i in enumerate(tensor):
    row_list.append(i)
    print(i)
print(torch.stack(row_list))
print(torch.cat(row_list))
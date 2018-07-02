require 'torch'

-- create a 2-D tensor of [2, 3] shape
a = torch.Tensor({{1.0,2.0,3.0},{4.0,5.0,6.0}})
-- create a 2-D tensor of [3, 2] shape
b = torch.Tensor({{1.0,2.0},{3.0,4.0},{5.0,6.0}})
-- perform a matrix product of those two tensors
c = torch.mm(a, b)

print('tensor a')
print(a)

print('tensor b')
print(b)

print('tensor c')
print(c)

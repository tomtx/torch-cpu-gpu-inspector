require 'torch'

-- a simple text
a = torch.Tensor({{1.0,2.0,3.0},{4.0,5.0,6.0}})
b = torch.Tensor({{1.0,2.0},{3.0,4.0},{5.0,6.0}})
-- tensor as matrix product between tensor a and b
c = torch.mm(a, b)

print('tensor a')
print(a)

print('tensor b')
print(b)

print('tensor c')
print(c)

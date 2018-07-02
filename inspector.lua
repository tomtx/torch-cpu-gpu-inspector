require 'torch'
require 'socket'

function test_cpu_gpu_computations()
    print("\n++++++++++++++++++++++++++++++")
    print("... placing matrix operations on available CPU/GPU devices")
    -- create a 2-D tensor of [2, 3] shape
    print("Tensor A")
    a = torch.Tensor({{1.0,2.0,3.0},{4.0,5.0,6.0}})
    print(a)
    -- create a 2-D tensor of [3, 2] shape
    print("Tensor B")
    b = torch.Tensor({{1.0,2.0},{3.0,4.0},{5.0,6.0}})
    print(b)
    -- perform a matrix product of those two tensors
    print("Matrix Product of Tensors A & B")
    c = torch.mm(a, b)
    print(c)
    print("++++++++++++++++++++++++++++++")
    return c
end

function main()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+++ Simple CPU/GPU Computation Test with Torch +++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("start:", os.date("%m/%d/%Y %I:%M %p"), socket.gettime()*1000, "ms")
    -- verify Torch installation
    test_cpu_gpu_computations()
    -- test CPU/GPU computations
    print("\nend:", os.date("%m/%d/%Y %I:%M %p"), socket.gettime()*1000, "ms")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
end

main()

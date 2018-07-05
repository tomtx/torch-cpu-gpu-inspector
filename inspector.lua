require 'torch'
require 'nn'
require 'socket'

function is_module_available(name)
  if package.loaded[name] then
    return true
  else
    for _, searcher in ipairs(package.searchers or package.loaders) do
      local loader = searcher(name)
      if type(loader) == 'function' then
        package.preload[name] = loader
        return true
      end
    end
    return false
  end
end

function perform_cpu_computations()
    print("\n++++++++++++++++++++++++++++++")
    print("... placing matrix operations on available CPU devices")
    -- create a 2-D tensor of [2, 3] shape
    print("CPU Tensor A")
    a = torch.Tensor({{1.0,2.0,3.0},{4.0,5.0,6.0}})
    print(a)
    -- create a 2-D tensor of [3, 2] shape
    print("CPU Tensor B")
    b = torch.Tensor({{1.0,2.0},{3.0,4.0},{5.0,6.0}})
    print(b)
    -- perform a matrix product of those two tensors
    print("Matrix Product of CPU Tensors A & B")
    start_time = socket.gettime()
    c = torch.mm(a, b)
    end_time = socket.gettime()
    elapsed_time = end_time - start_time
    print(c)
    print("... time to finish the workload: %f s" % elapsed_time)
    print("\n... matrix operations has been completed")
    print("++++++++++++++++++++++++++++++")
end

function perform_gpu_computations()
    print("\n++++++++++++++++++++++++++++++")
    print("... placing matrix operations on available GPU devices")
    cutorch = require 'cutorch'
    -- create a 2-D tensor of [2, 3] shape
    print("GPU Tensor A")
    a = torch.CudaTensor({{1.0,2.0,3.0},{4.0,5.0,6.0}})
    print(a)
    -- create a 2-D tensor of [3, 2] shape
    print("GPU Tensor B")
    b = torch.CudaTensor({{1.0,2.0},{3.0,4.0},{5.0,6.0}})
    print(b)
    -- perform a matrix product of those two tensors
    print("Matrix Product of GPU Tensors A & B")
    start_time = socket.gettime()
    c = torch.mm(a, b)
    end_time = socket.gettime()
    elapsed_time = end_time - start_time
    print(c)
    print("... time to finish the workload: %f s" % elapsed_time)
    print("\n... matrix operations has been completed")
    print("++++++++++++++++++++++++++++++")
end

function test_cpu_gpu_computations()
    -- Perform CPU computations
    perform_cpu_computations()
    -- Perform GPU computations
    -- check if the GPU libraries are present
    gpu_libraries = is_module_available('cutorch')
    if gpu_libraries then
        perform_gpu_computations()
    else
        print("\n++++++++++++++++++++++++++++++")
        print("... GPU devices or libraries are not present!")
        print("++++++++++++++++++++++++++++++")
    end
end

function verify_torch_installation()
    print("\n++++++++++++++++++++++++++++++")
    print("... verifying Torch installation with its built-in tests")
    torch.test()
    nn.test()
    print("... verification of Torch installation has been completed")
    print("++++++++++++++++++++++++++++++")
end

function main()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+++ Simple CPU/GPU Computation Test with Torch +++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("start:", os.date("%m/%d/%Y %I:%M %p"), socket.gettime()*1000, "ms")
    -- verify Torch installation
    verify_torch_installation()
    -- test CPU/GPU computations
    test_cpu_gpu_computations()
    print("\nend:", os.date("%m/%d/%Y %I:%M %p"), socket.gettime()*1000, "ms")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
end

main()

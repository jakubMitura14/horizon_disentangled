using PackageCompiler
using Pkg

# Setup environment
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Define packages to precompile
packages = [:Lux, :MPI, :Zygote, :Optimisers, :CUDA, :ComponentArrays, :Statistics, :Random, :Printf, :ArgParse]

# Create system image
# We trace a small run to ensure JIT actions are captured
# But strictly, just including the packages helps load time.
# For full benefit, we should run the `train_lux_distributed.jl` with small epochs during build.

create_sysimage(packages;
    sysimage_path=joinpath(@__DIR__, "lib/sysimage.so"),
    precompile_execution_file=joinpath(@__DIR__, "src/train_lux_distributed.jl"), # Dry run
    script=joinpath(@__DIR__, "src/train_lux_distributed.jl") # Or just the entry point
)

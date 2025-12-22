using Lux
using Random
using NNlib

# --- Common Layers ---

struct ResNetBlock{L1, L2} <: Lux.AbstractLuxLayer
    layer1::L1
    layer2::L2
end

function ResNetBlock(in_channels, out_channels)
    layer1 = Chain(
        Conv((3, 3, 3), in_channels => out_channels, pad=1),
        InstanceNorm(out_channels, relu)
    )
    layer2 = Chain(
        Conv((3, 3, 3), out_channels => out_channels, pad=1),
        InstanceNorm(out_channels, relu)
    )
    return ResNetBlock(layer1, layer2)
end

function Lux.initialparameters(rng::AbstractRNG, m::ResNetBlock)
    return (layer1 = Lux.initialparameters(rng, m.layer1),
            layer2 = Lux.initialparameters(rng, m.layer2))
end

function Lux.initialstates(rng::AbstractRNG, m::ResNetBlock)
    return (layer1 = Lux.initialstates(rng, m.layer1),
            layer2 = Lux.initialstates(rng, m.layer2))
end

function (m::ResNetBlock)(x, ps, st)
    y, st1 = m.layer1(x, ps.layer1, st.layer1)
    y, st2 = m.layer2(y, ps.layer2, st.layer2)
    # Skip connection (simplified, assumes dimensions match or handling externally)
    return y + x, (layer1=st1, layer2=st2)
end

# --- SPADE Layer ---
struct SPADE{L1, L2, L3} <: Lux.AbstractLuxLayer
    norm::InstanceNorm
    shared::L1
    gamma::L2
    beta::L3
end

function SPADE(norm_nc, label_nc)
    return SPADE(
        InstanceNorm(norm_nc, affine=false),
        Conv((3,3,3), label_nc => 128, pad=1, relu),
        Conv((3,3,3), 128 => norm_nc, pad=1),
        Conv((3,3,3), 128 => norm_nc, pad=1)
    )
end

function Lux.initialparameters(rng::AbstractRNG, m::SPADE)
    return (norm = Lux.initialparameters(rng, m.norm),
            shared = Lux.initialparameters(rng, m.shared),
            gamma = Lux.initialparameters(rng, m.gamma),
            beta = Lux.initialparameters(rng, m.beta))
end

function Lux.initialstates(rng::AbstractRNG, m::SPADE)
    return (norm = Lux.initialstates(rng, m.norm),
            shared = Lux.initialstates(rng, m.shared),
            gamma = Lux.initialstates(rng, m.gamma),
            beta = Lux.initialstates(rng, m.beta))
end

function (m::SPADE)(x, segmap, ps, st)
    normalized, st_norm = m.norm(x, ps.norm, st.norm)
    actv, st_shared = m.shared(segmap, ps.shared, st.shared)
    gamma, st_gamma = m.gamma(actv, ps.gamma, st.gamma)
    beta, st_beta = m.beta(actv, ps.beta, st.beta)

    out = normalized .* (1 .+ gamma) .+ beta
    return out, (norm=st_norm, shared=st_shared, gamma=st_gamma, beta=st_beta)
end

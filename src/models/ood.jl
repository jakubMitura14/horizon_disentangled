using Lux
using Random

function OODDetector(dim=16, latent=4)
    # Simple VAE on latent space
    encoder = Chain(Dense(dim => 12, relu), Dense(12 => latent * 2))
    decoder = Chain(Dense(latent => 12, relu), Dense(12 => dim))
    return Chain(encoder=encoder, decoder=decoder)
end

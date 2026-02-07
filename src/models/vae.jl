using Lux
using Random
using Statistics
using Zygote
using NNlib

include(joinpath(@__DIR__, "layers.jl"))

# --- Gaussian Encoder (Struct based) ---
struct GaussianEncoder{E, M, L} <: Lux.AbstractLuxContainerLayer{(:embed, :proj_mu, :proj_log_var)}
    embed::E
    proj_mu::M
    proj_log_var::L
end

function GaussianEncoder(in_channels, latent_dim)
    embed = Chain(
        Conv((3,3,3), in_channels => 16, stride=2, pad=1, relu),
        Conv((3,3,3), 16 => 32, stride=2, pad=1, relu),
        GlobalMaxPool(),
        FlattenLayer()
    )
    # Input after GlobalMaxPool on 32 filters is 32.
    proj_mu = Dense(32 => latent_dim)
    proj_log_var = Dense(32 => latent_dim)
    
    return GaussianEncoder(embed, proj_mu, proj_log_var)
end

function (m::GaussianEncoder)(x, ps, st)
    y, st_embed = m.embed(x, ps.embed, st.embed)
    mu, st_mu = m.proj_mu(y, ps.proj_mu, st.proj_mu)
    logσ², st_log = m.proj_log_var(y, ps.proj_log_var, st.proj_log_var)
    
    # Clamp for stability
    logσ² = clamp.(logσ², -20.0f0, 10.0f0)
    σ = exp.(logσ² .* 0.5f0)
    
    # Reparameterization
    eps = Zygote.ignore() do
        randn!(similar(σ))
    end
    
    z = mu .+ σ .* eps
    
    return (z, mu, logσ²), (embed=st_embed, proj_mu=st_mu, proj_log_var=st_log)
end

function AnatomyEncoder(in_channels=1, out_channels=4)
    return Chain(
        Conv((3,3,3), in_channels => 16, pad=1, relu),
        Conv((3,3,3), 16 => out_channels, pad=1, relu)
    )
end

# --- VAE Model ---
struct CausalVAE{E1, E2, E3, D1, D2} <: Lux.AbstractLuxContainerLayer{(:enc_p, :enc_s, :enc_a, :decoder_start, :decoder_conv)}
    enc_p::E1
    enc_s::E2
    enc_a::E3
    decoder_start::D1
    decoder_conv::D2
end

function CausalVAE(latent_dim=16, anatomy_dim=4)
    enc_p = GaussianEncoder(2, latent_dim)
    enc_s = GaussianEncoder(2, latent_dim)
    enc_a = AnatomyEncoder(1, anatomy_dim)

    decoder_start = Dense(latent_dim*2 => 16 * 6 * 6 * 2) 
    
    # Decoder with upsampling 6x6 -> 12x12 -> 24x24 -> 48x48
    decoder_conv = Chain(
        ConvTranspose((4,4,4), 16 => 8, stride=2, pad=1, relu), # 6->12
        ConvTranspose((4,4,4), 8 => 8, stride=2, pad=1, relu),  # 12->24
        ConvTranspose((4,4,4), 8 => 4, stride=2, pad=1, relu),  # 24->48
        Conv((3,3,3), 4 => 1, pad=1)                            # Final 48x48x16, 1 channel
    )

    return CausalVAE(enc_p, enc_s, enc_a, decoder_start, decoder_conv)
end

function (m::CausalVAE)(inputs, ps, st)
    img, mask = inputs

    # Encode
    (z_p, mu_p, log_p), st_p = m.enc_p(img, ps.enc_p, st.enc_p)
    (z_s, mu_s, log_s), st_s = m.enc_s(img, ps.enc_s, st.enc_s)
    
    # Anatomy (Spatial)
    s, st_a = m.enc_a(mask, ps.enc_a, st.enc_a)

    # Decode
    z = vcat(z_p, z_s) # Concatenate latents
    recon, st_dec = decode(m, z, ps, st)

    # Return
    return (recon, mu_p, log_p, mu_s, log_s),
           (enc_p=st_p, enc_s=st_s, enc_a=st_a, decoder_start=st_dec.decoder_start, decoder_conv=st_dec.decoder_conv)
end

function decode(m::CausalVAE, z, ps, st)
    h, st_dstart = m.decoder_start(z, ps.decoder_start, st.decoder_start)

    # Reshape
    # (B, 16*6*6*2) -> (6, 6, 2, 16, B)
    batch = size(z, 2)
    h_reshaped = reshape(h, (6, 6, 2, 16, batch))

    recon, st_dconv = m.decoder_conv(h_reshaped, ps.decoder_conv, st.decoder_conv)
    
    return recon, (decoder_start=st_dstart, decoder_conv=st_dconv)
end

function generate(m::CausalVAE, ps, st; num_samples=1, latent_dim=16, device=cpu_device(), rng=Random.default_rng())
    # Sample from standard normal prior
    # z vector is concatenation of z_p and z_s, both size latent_dim
    # Total size = latent_dim * 2
    z = randn(rng, Float32, latent_dim * 2, num_samples) |> device
    
    # Decode
    recon, _ = decode(m, z, ps, st)
    return recon
end

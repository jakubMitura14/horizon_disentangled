# Merlin Report Generator - Combines I3ResNet Image Encoder with GPT-2 Text Decoder
# for generating radiological reports from 3D medical images

using Lux
using Random

include("merlin.jl")  # I3ResNet
include("gpt2.jl")    # GPT-2

# ============================================================================
# Adapter Layer (Projects image features to text decoder dimension)
# ============================================================================
struct ImageTextAdapter{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

function ImageTextAdapter(in_dim::Int, out_dim::Int)
    return ImageTextAdapter(Dense(in_dim, out_dim))
end

function (l::ImageTextAdapter)(x, ps, st)
    # x: (in_dim, seq_len, batch) - image features
    out, st_l = l.linear(x, ps.linear, st.linear)
    return out, (linear=st_l,)
end

# ============================================================================
# Modified I3ResNet that returns spatial features instead of pooled
# ============================================================================
struct I3ResNetFeatures{C,B,M,L1,L2,L3,L4} <: Lux.AbstractLuxContainerLayer{(:conv1, :bn1, :maxpool, :layer1, :layer2, :layer3, :layer4)}
    conv1::C
    bn1::B
    maxpool::M
    layer1::L1
    layer2::L2
    layer3::L3
    layer4::L4
end

function I3ResNetFeatures()
    # Same as I3ResNet but without avgpool and fc
    # Returns (2048, D', H', W', batch) spatial features
    
    # Initial convolution
    conv1 = Conv((7, 7, 3), 3 => 64; stride=(2, 2, 1), pad=(3, 3, 1))
    bn1 = BatchNorm(64)
    maxpool = MaxPool((3, 3, 3); stride=(2, 2, 2), pad=(1, 1, 1))
    
    # ResNet layers (same as I3ResNet)
    layer1 = make_layer(64, 64, 256, 3, stride=1)
    layer2 = make_layer(256, 128, 512, 8, stride=2)
    layer3 = make_layer(512, 256, 1024, 36, stride=2)
    layer4 = make_layer(1024, 512, 2048, 3, stride=2)
    
    return I3ResNetFeatures(conv1, bn1, maxpool, layer1, layer2, layer3, layer4)
end

function (l::I3ResNetFeatures)(x, ps, st)
    # x: (W, H, D, C, N)
    # Triplicate channel if needed
    if size(x, 4) == 1
        x_tri = cat(x, x, x; dims=4)
    else
        x_tri = x
    end
    
    y, st_c1 = l.conv1(x_tri, ps.conv1, st.conv1)
    y, st_bn1 = l.bn1(y, ps.bn1, st.bn1)
    y = relu.(y)
    y, st_mp = l.maxpool(y, ps.maxpool, st.maxpool)
    
    y, st_l1 = l.layer1(y, ps.layer1, st.layer1)
    y, st_l2 = l.layer2(y, ps.layer2, st.layer2)
    y, st_l3 = l.layer3(y, ps.layer3, st.layer3)
    y, st_l4 = l.layer4(y, ps.layer4, st.layer4)
    
    # y: (W', H', D', 2048, N) - spatial features
    new_st = (conv1=st_c1, bn1=st_bn1, maxpool=st_mp, 
              layer1=st_l1, layer2=st_l2, layer3=st_l3, layer4=st_l4)
    
    return y, new_st
end

# ============================================================================
# Merlin Report Generator
# ============================================================================
struct MerlinReportGenerator{E,A,D} <: Lux.AbstractLuxContainerLayer{(:image_encoder, :adapter, :text_decoder)}
    image_encoder::E
    adapter::A
    text_decoder::D
    config::GPT2Config
end

function MerlinReportGenerator(gpt2_config::GPT2Config=GPT2Config())
    # I3ResNet outputs 2048-dim features
    # GPT-2 expects 768-dim (for GPT-2 small) or 4096 for LLaMA
    # Use adapter to project
    
    image_encoder = I3ResNet()  # Use existing I3ResNet
    adapter = ImageTextAdapter(2048, gpt2_config.n_embd)
    text_decoder = GPT2LMHead(gpt2_config)
    
    return MerlinReportGenerator(image_encoder, adapter, text_decoder, gpt2_config)
end

function encode_image(model::MerlinReportGenerator, image, ps, st)
    # Encode image to embeddings
    # image: (W, H, D, C, N)
    
    # Get image features
    img_features, st_enc = model.image_encoder(image, ps.image_encoder, st.image_encoder)
    
    # img_features: (W', H', D', 2048, N)
    # Flatten spatial dims: (2048, W'*H'*D', N)
    w, h, d, c, n = size(img_features)
    img_flat = reshape(img_features, w * h * d, c, n)  # (W'*H'*D', 2048, N)
    img_flat = permutedims(img_flat, (2, 1, 3))  # (2048, seq, N)
    
    # Project to text decoder dimension
    img_emb, st_adp = model.adapter(img_flat, ps.adapter, st.adapter)
    # img_emb: (n_embd, seq, N)
    
    new_st = (image_encoder=st_enc, adapter=st_adp, text_decoder=st.text_decoder)
    return img_emb, new_st
end

function generate_report(model::MerlinReportGenerator, image, tokenizer, ps, st;
                         max_new_tokens::Int=200, prompt::String="Findings:")
    # Generate radiological report from image
    
    # 1. Encode image
    img_emb, st = encode_image(model, image, ps, st)
    # img_emb: (n_embd, img_seq_len, batch)
    
    _, img_seq_len, batch = size(img_emb)
    
    # 2. Encode prompt
    prompt_ids = encode(tokenizer, prompt)
    prompt_ids = reshape(prompt_ids, length(prompt_ids), 1)  # (seq, 1)
    
    # Get prompt embeddings using text decoder's wte
    prompt_emb, _ = model.text_decoder.transformer.wte(prompt_ids, 
        ps.text_decoder.transformer.wte, st.text_decoder.transformer.wte)
    # prompt_emb: (n_embd, prompt_len, 1)
    
    # Concatenate image and prompt embeddings
    if batch == 1
        combined_emb = cat(img_emb, prompt_emb; dims=2)
    else
        # Repeat prompt for batch
        prompt_emb_batch = repeat(prompt_emb, 1, 1, batch)
        combined_emb = cat(img_emb, prompt_emb_batch; dims=2)
    end
    # combined_emb: (n_embd, img_seq + prompt_len, batch)
    
    # 3. Autoregressive generation
    generated_ids = Int[]
    current_emb = combined_emb
    
    for _ in 1:max_new_tokens
        # Get position ids for the sequence
        seq_len = size(current_emb, 2)
        pos_ids = collect(1:seq_len)
        
        # Get positional embeddings
        pos_emb, _ = model.text_decoder.transformer.wpe(pos_ids,
            ps.text_decoder.transformer.wpe, st.text_decoder.transformer.wpe)
        
        # Add positional embeddings
        hidden = current_emb .+ pos_emb
        
        # Pass through transformer blocks
        hidden, st_h = model.text_decoder.transformer.h(hidden, 
            ps.text_decoder.transformer.h, st.text_decoder.transformer.h)
        
        # Final layer norm
        hidden, st_ln = model.text_decoder.transformer.ln_f(hidden,
            ps.text_decoder.transformer.ln_f, st.text_decoder.transformer.ln_f)
        
        # Get logits for last position
        last_hidden = hidden[:, end:end, :]  # (n_embd, 1, batch)
        
        # Project to vocabulary
        wte_weight = ps.text_decoder.transformer.wte.weight  # (n_embd, vocab_size)
        logits = permutedims(wte_weight, (2, 1)) * last_hidden[:, 1, 1]  # (vocab_size,)
        
        # Greedy decoding
        next_token_id = argmax(logits)
        push!(generated_ids, next_token_id)
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id
            break
        end
        
        # Get embedding for next token and append
        next_ids = reshape([next_token_id], 1, 1)
        next_emb, _ = model.text_decoder.transformer.wte(next_ids,
            ps.text_decoder.transformer.wte, st.text_decoder.transformer.wte)
        
        current_emb = cat(current_emb, next_emb; dims=2)
    end
    
    # 4. Decode to text
    report_text = decode(tokenizer, generated_ids)
    
    return prompt * report_text
end

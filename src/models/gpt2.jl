# GPT-2 Decoder implementation in Lux.jl for Merlin text generation
# This implements GPT-2 Small (124M params, 12 layers, 768 hidden, 12 heads)

using Lux
using Random
using NNlib

# ============================================================================
# GPT-2 Configuration
# ============================================================================
struct GPT2Config
    vocab_size::Int      # 50257 for GPT-2
    n_positions::Int     # 1024 max sequence length
    n_embd::Int          # 768 hidden size
    n_layer::Int         # 12 transformer blocks
    n_head::Int          # 12 attention heads
    n_inner::Int         # 3072 (4 * n_embd) FFN intermediate size
    layer_norm_epsilon::Float32
end

function GPT2Config()
    return GPT2Config(
        50257,    # vocab_size
        1024,     # n_positions
        768,      # n_embd
        12,       # n_layer
        12,       # n_head
        3072,     # n_inner
        1e-5f0    # layer_norm_epsilon
    )
end

# ============================================================================
# Causal Self-Attention
# ============================================================================
struct CausalSelfAttention{Q,K,V,P} <: Lux.AbstractLuxContainerLayer{(:q_proj, :k_proj, :v_proj, :out_proj)}
    q_proj::Q
    k_proj::K
    v_proj::V
    out_proj::P
    n_head::Int
    head_dim::Int
end

function CausalSelfAttention(n_embd::Int, n_head::Int)
    head_dim = n_embd ÷ n_head
    return CausalSelfAttention(
        Dense(n_embd, n_embd),      # q_proj (c_attn.weight[:, :n_embd])
        Dense(n_embd, n_embd),      # k_proj
        Dense(n_embd, n_embd),      # v_proj
        Dense(n_embd, n_embd),      # out_proj (c_proj)
        n_head,
        head_dim
    )
end

function (l::CausalSelfAttention)(x, ps, st)
    # x: (n_embd, seq_len, batch)
    n_embd, seq_len, batch = size(x)
    
    # Project Q, K, V
    q, st_q = l.q_proj(x, ps.q_proj, st.q_proj)  # (n_embd, seq_len, batch)
    k, st_k = l.k_proj(x, ps.k_proj, st.k_proj)
    v, st_v = l.v_proj(x, ps.v_proj, st.v_proj)
    
    # Reshape to (head_dim, n_head, seq_len, batch)
    q = reshape(q, l.head_dim, l.n_head, seq_len, batch)
    k = reshape(k, l.head_dim, l.n_head, seq_len, batch)
    v = reshape(v, l.head_dim, l.n_head, seq_len, batch)
    
    # Compute attention scores: (seq_len, seq_len, n_head, batch)
    # attn = softmax(Q @ K^T / sqrt(d_k) + causal_mask)
    scale = Float32(1.0 / sqrt(l.head_dim))
    
    # Batch matrix multiply: for each head and batch
    # We need attn[i,j] = q[:,i] . k[:,j]
    # q: (head_dim, n_head, seq_len, batch) -> permute to (head_dim, seq_len, n_head, batch)
    q_p = permutedims(q, (1, 3, 2, 4))  # (head_dim, seq_len_q, n_head, batch)
    k_p = permutedims(k, (1, 3, 2, 4))  # (head_dim, seq_len_k, n_head, batch)
    v_p = permutedims(v, (1, 3, 2, 4))  # (head_dim, seq_len_v, n_head, batch)
    
    # Reshape for batched matmul: (head_dim, seq_len, n_head*batch)
    q_flat = reshape(q_p, l.head_dim, seq_len, l.n_head * batch)
    k_flat = reshape(k_p, l.head_dim, seq_len, l.n_head * batch)
    v_flat = reshape(v_p, l.head_dim, seq_len, l.n_head * batch)
    
    # Compute attention: Q @ K^T -> (seq_len, seq_len, n_head*batch)
    attn_scores = NNlib.batched_mul(
        permutedims(q_flat, (2, 1, 3)),  # (seq_len, head_dim, n_head*batch)
        k_flat                            # (head_dim, seq_len, n_head*batch)
    ) .* scale  # (seq_len_q, seq_len_k, n_head*batch)
    
    # Apply causal mask (upper triangular = -inf)
    causal_mask = Float32.(triu(fill(-Inf32, seq_len, seq_len), 1))
    attn_scores = attn_scores .+ causal_mask
    
    # Softmax over key dimension (dim 2)
    attn_probs = softmax(attn_scores, dims=2)
    
    # Apply attention to values: attn @ V -> (seq_len, head_dim, n_head*batch)
    attn_out = NNlib.batched_mul(
        attn_probs,                                    # (seq_len, seq_len, n_head*batch)
        permutedims(v_flat, (2, 1, 3))                # (seq_len, head_dim, n_head*batch)
    )  # (seq_len, head_dim, n_head*batch)
    
    # Reshape back: (seq_len, head_dim, n_head, batch) -> (head_dim, n_head, seq_len, batch)
    attn_out = reshape(attn_out, seq_len, l.head_dim, l.n_head, batch)
    attn_out = permutedims(attn_out, (2, 3, 1, 4))  # (head_dim, n_head, seq_len, batch)
    
    # Concatenate heads: (n_embd, seq_len, batch)
    attn_out = reshape(attn_out, l.head_dim * l.n_head, seq_len, batch)
    
    # Output projection
    out, st_o = l.out_proj(attn_out, ps.out_proj, st.out_proj)
    
    new_st = (q_proj=st_q, k_proj=st_k, v_proj=st_v, out_proj=st_o)
    return out, new_st
end

# ============================================================================
# GPT-2 MLP (Feed-Forward Network)
# ============================================================================
struct GPT2MLP{F,P} <: Lux.AbstractLuxContainerLayer{(:fc, :proj)}
    fc::F
    proj::P
end

function GPT2MLP(n_embd::Int, n_inner::Int)
    return GPT2MLP(
        Dense(n_embd, n_inner, gelu),  # c_fc
        Dense(n_inner, n_embd)         # c_proj
    )
end

function (l::GPT2MLP)(x, ps, st)
    h, st_fc = l.fc(x, ps.fc, st.fc)
    out, st_proj = l.proj(h, ps.proj, st.proj)
    return out, (fc=st_fc, proj=st_proj)
end

# ============================================================================
# GPT-2 Block (Transformer Block)
# ============================================================================
struct GPT2Block{LN1,A,LN2,M} <: Lux.AbstractLuxContainerLayer{(:ln_1, :attn, :ln_2, :mlp)}
    ln_1::LN1
    attn::A
    ln_2::LN2
    mlp::M
end

function GPT2Block(config::GPT2Config)
    return GPT2Block(
        LayerNorm((config.n_embd,), epsilon=config.layer_norm_epsilon),
        CausalSelfAttention(config.n_embd, config.n_head),
        LayerNorm((config.n_embd,), epsilon=config.layer_norm_epsilon),
        GPT2MLP(config.n_embd, config.n_inner)
    )
end

function (l::GPT2Block)(x, ps, st)
    # Pre-LN Transformer: x + attn(ln1(x)), x + mlp(ln2(x))
    
    # Self-attention with residual
    h_ln1, st_ln1 = l.ln_1(x, ps.ln_1, st.ln_1)
    h_attn, st_attn = l.attn(h_ln1, ps.attn, st.attn)
    x = x .+ h_attn
    
    # MLP with residual
    h_ln2, st_ln2 = l.ln_2(x, ps.ln_2, st.ln_2)
    h_mlp, st_mlp = l.mlp(h_ln2, ps.mlp, st.mlp)
    x = x .+ h_mlp
    
    new_st = (ln_1=st_ln1, attn=st_attn, ln_2=st_ln2, mlp=st_mlp)
    return x, new_st
end

# ============================================================================
# GPT-2 Model (Full Decoder Stack)
# ============================================================================
struct GPT2Model{WTE,WPE,B,LNF} <: Lux.AbstractLuxContainerLayer{(:wte, :wpe, :h, :ln_f)}
    wte::WTE      # Token embeddings
    wpe::WPE      # Positional embeddings
    h::B          # Transformer blocks
    ln_f::LNF     # Final layer norm
    config::GPT2Config
end

function GPT2Model(config::GPT2Config=GPT2Config())
    blocks = [GPT2Block(config) for _ in 1:config.n_layer]
    return GPT2Model(
        Embedding(config.vocab_size => config.n_embd),  # wte
        Embedding(config.n_positions => config.n_embd), # wpe
        Chain(blocks...),                                # h
        LayerNorm((config.n_embd,), epsilon=config.layer_norm_epsilon),
        config
    )
end

function (l::GPT2Model)(input_ids, ps, st)
    # input_ids: (seq_len, batch) - integer token IDs
    seq_len, batch = size(input_ids)
    
    # Token embeddings: (n_embd, seq_len, batch)
    tok_emb, st_wte = l.wte(input_ids, ps.wte, st.wte)
    
    # Positional embeddings
    positions = collect(1:seq_len)  # (seq_len,)
    pos_emb, st_wpe = l.wpe(positions, ps.wpe, st.wpe)  # (n_embd, seq_len)
    
    # Combine: (n_embd, seq_len, batch)
    x = tok_emb .+ pos_emb
    
    # Through transformer blocks
    x, st_h = l.h(x, ps.h, st.h)
    
    # Final layer norm
    x, st_ln = l.ln_f(x, ps.ln_f, st.ln_f)
    
    new_st = (wte=st_wte, wpe=st_wpe, h=st_h, ln_f=st_ln)
    return x, new_st  # (n_embd, seq_len, batch)
end

# ============================================================================
# GPT-2 LM Head (for text generation)
# ============================================================================
struct GPT2LMHead{M} <: Lux.AbstractLuxContainerLayer{(:transformer,)}
    transformer::M
    config::GPT2Config
end

function GPT2LMHead(config::GPT2Config=GPT2Config())
    return GPT2LMHead(GPT2Model(config), config)
end

function (l::GPT2LMHead)(input_ids, ps, st)
    # Get hidden states
    hidden_states, st_t = l.transformer(input_ids, ps.transformer, st.transformer)
    
    # Project to vocabulary using tied weights (wte weights)
    # hidden_states: (n_embd, seq_len, batch)
    # wte.weight: (n_embd, vocab_size)
    
    # Logits = hidden @ wte^T -> (vocab_size, seq_len, batch)
    wte_weight = ps.transformer.wte.weight  # (n_embd, vocab_size)
    
    # Batched matmul: (vocab_size, n_embd) @ (n_embd, seq_len, batch)
    # For each position in batch:
    logits = NNlib.batched_mul(
        permutedims(wte_weight, (2, 1)),  # (vocab_size, n_embd)
        hidden_states                      # (n_embd, seq_len, batch)
    )  # Will broadcast incorrectly...
    
    # Actually need to do this manually for each batch
    n_embd, seq_len, batch = size(hidden_states)
    logits = reshape(
        permutedims(wte_weight, (2, 1)) * reshape(hidden_states, n_embd, seq_len * batch),
        l.config.vocab_size, seq_len, batch
    )
    
    return logits, (transformer=st_t,)
end

# ============================================================================
# Text Generation (Greedy Decoding)
# ============================================================================
function generate_greedy(model, ps, st, input_ids::AbstractMatrix{<:Integer}; 
                         max_new_tokens::Int=100, eos_token_id::Int=50256)
    generated = copy(input_ids)
    
    for _ in 1:max_new_tokens
        # Get logits for current sequence
        logits, st = model(generated, ps, st)
        
        # Get next token (greedy: argmax of last position)
        next_logits = logits[:, end, :]  # (vocab_size, batch)
        next_tokens = argmax(next_logits, dims=1)  # (1, batch)
        next_tokens = dropdims(next_tokens, dims=1)  # (batch,)
        
        # Convert CartesianIndex to Int
        next_ids = [idx[1] for idx in next_tokens]
        next_ids = reshape(next_ids, 1, length(next_ids))  # (1, batch)
        
        # Append
        generated = vcat(generated, next_ids)
        
        # Check for EOS
        if all(next_ids .== eos_token_id)
            break
        end
    end
    
    return generated
end

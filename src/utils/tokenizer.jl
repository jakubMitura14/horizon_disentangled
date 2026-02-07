# BPE Tokenizer implementation for GPT-2 in pure Julia
# Loads vocab.json and merges.txt from HuggingFace GPT-2 tokenizer

using JSON

# ============================================================================
# BPE Tokenizer
# ============================================================================
struct BPETokenizer
    encoder::Dict{String, Int}        # token -> id
    decoder::Dict{Int, String}        # id -> token
    bpe_ranks::Dict{Tuple{String, String}, Int}  # BPE merge ranks
    byte_encoder::Dict{UInt8, Char}   # Byte -> Unicode char mapping
    byte_decoder::Dict{Char, UInt8}   # Unicode char -> Byte mapping
    pat::Regex                         # Pattern for tokenization
    eos_token_id::Int
    bos_token_id::Int
    pad_token_id::Int
end

function bytes_to_unicode()
    # GPT-2 uses a byte-level BPE that maps bytes to Unicode characters
    bs = UInt8[]
    for i in UInt8('!'):UInt8('~')
        push!(bs, i)
    end
    for i in 0xa1:0xac  # extended latin
        push!(bs, UInt8(i))
    end
    for i in 0xae:0xff  # extended latin
        push!(bs, UInt8(i))
    end
    
    cs = copy(bs)
    n = 0
    for b in 0x00:0xff
        if !(UInt8(b) in bs)
            push!(bs, UInt8(b))
            push!(cs, UInt8(256 + n))
            n += 1
        end
    end
    
    cs_chars = [Char(c) for c in cs]
    return Dict(zip(bs, cs_chars))
end

function BPETokenizer(vocab_path::String, merges_path::String)
    # Load vocab
    encoder = JSON.parsefile(vocab_path)
    encoder = Dict{String, Int}(String(k) => v for (k, v) in encoder)
    decoder = Dict{Int, String}(v => k for (k, v) in encoder)
    
    # Load merges
    merges_text = read(merges_path, String)
    merges_lines = split(merges_text, '\n')
    
    # Skip first line (version header) and filter empty lines
    merge_pairs = Tuple{String, String}[]
    for line in merges_lines[2:end]
        stripped = strip(line)
        if !isempty(stripped)
            parts = split(stripped, ' ')
            if length(parts) == 2
                push!(merge_pairs, (String(parts[1]), String(parts[2])))
            end
        end
    end
    
    bpe_ranks = Dict{Tuple{String, String}, Int}()
    for (i, pair) in enumerate(merge_pairs)
        bpe_ranks[pair] = i
    end
    
    # Byte encoder/decoder
    byte_encoder = bytes_to_unicode()
    byte_decoder = Dict(v => k for (k, v) in byte_encoder)
    
    # GPT-2 tokenization pattern
    pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    
    # Special tokens (GPT-2 uses 50256 as EOS which is the endoftext token)
    endoftext = string(Char(60), "endoftext", Char(62))  # angle bracket endoftext angle bracket
    eos_token_id = get(encoder, endoftext, 50256)
    bos_token_id = eos_token_id  # GPT-2 uses same token for BOS/EOS
    pad_token_id = eos_token_id
    
    return BPETokenizer(
        encoder, decoder, bpe_ranks, 
        byte_encoder, byte_decoder, 
        pat, eos_token_id, bos_token_id, pad_token_id
    )
end

# ============================================================================
# BPE Algorithm
# ============================================================================
function get_pairs(word::Vector{String})
    pairs = Set{Tuple{String, String}}()
    for i in 1:(length(word) - 1)
        push!(pairs, (word[i], word[i+1]))
    end
    return pairs
end

function bpe(tokenizer::BPETokenizer, token::String)
    # Convert token to BPE representation
    word = [string(c) for c in token]
    
    if length(word) <= 1
        return word
    end
    
    while true
        pairs = get_pairs(word)
        if isempty(pairs)
            break
        end
        
        # Find the pair with the lowest rank
        best_pair = nothing
        best_rank = typemax(Int)
        for pair in pairs
            rank = get(tokenizer.bpe_ranks, pair, typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_pair = pair
            end
        end
        
        if best_pair === nothing || best_rank == typemax(Int)
            break
        end
        
        # Merge the pair
        first, second = best_pair
        new_word = String[]
        i = 1
        while i <= length(word)
            if i < length(word) && word[i] == first && word[i+1] == second
                push!(new_word, first * second)
                i += 2
            else
                push!(new_word, word[i])
                i += 1
            end
        end
        word = new_word
        
        if length(word) == 1
            break
        end
    end
    
    return word
end

# ============================================================================
# Encode / Decode
# ============================================================================
function encode(tokenizer::BPETokenizer, text::String)
    # Tokenize text to token IDs
    token_ids = Int[]
    
    # Find all matches using the pattern
    for m in eachmatch(tokenizer.pat, text)
        token = m.match
        
        # Convert to byte-level representation
        byte_token = join([tokenizer.byte_encoder[b] for b in Vector{UInt8}(token)])
        
        # Apply BPE
        bpe_tokens = bpe(tokenizer, byte_token)
        
        # Convert to IDs
        for bt in bpe_tokens
            if haskey(tokenizer.encoder, bt)
                push!(token_ids, tokenizer.encoder[bt])
            else
                # Unknown token - should not happen with byte-level BPE
                @warn "Unknown token: $bt"
            end
        end
    end
    
    return token_ids
end

function decode(tokenizer::BPETokenizer, token_ids::Vector{Int})
    # Decode token IDs to text
    tokens = [get(tokenizer.decoder, id, "") for id in token_ids]
    text = join(tokens)
    
    # Convert back from byte-level
    bytes = UInt8[]
    for c in text
        if haskey(tokenizer.byte_decoder, c)
            push!(bytes, tokenizer.byte_decoder[c])
        end
    end
    
    return String(bytes)
end

# ============================================================================
# Download GPT-2 Tokenizer Files
# ============================================================================
function download_gpt2_tokenizer(output_dir::String)
    mkpath(output_dir)
    
    vocab_url = "https://huggingface.co/gpt2/resolve/main/vocab.json"
    merges_url = "https://huggingface.co/gpt2/resolve/main/merges.txt"
    
    vocab_path = joinpath(output_dir, "vocab.json")
    merges_path = joinpath(output_dir, "merges.txt")
    
    if !isfile(vocab_path)
        println("Downloading vocab.json...")
        download(vocab_url, vocab_path)
    end
    
    if !isfile(merges_path)
        println("Downloading merges.txt...")
        download(merges_url, merges_path)
    end
    
    return vocab_path, merges_path
end

function load_gpt2_tokenizer(tokenizer_dir::String)
    vocab_path = joinpath(tokenizer_dir, "vocab.json")
    merges_path = joinpath(tokenizer_dir, "merges.txt")
    
    if !isfile(vocab_path) || !isfile(merges_path)
        download_gpt2_tokenizer(tokenizer_dir)
    end
    
    return BPETokenizer(vocab_path, merges_path)
end

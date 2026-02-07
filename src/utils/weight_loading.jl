using SafeTensors
using Lux
using ComponentArrays


"""
    load_totalseg_weights(model, safetensors_path)

Load weights from a SafeTensors file into a Lux model.
handles permutation of Conv3d weights from PyTorch (N, C, D, H, W) to Lux (W, H, D, C, N).
"""
function load_totalseg_weights(model, safetensors_path)
    # Load raw tensors
    tensors = SafeTensors.load_safetensors(safetensors_path)
    
    # Create initial parameters to get the structure
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    ps_flat = ComponentArray(ps)
    
    # We need to map keys.
    # PyTorch keys look like: 
    # encoder.stages.0.layers.0.conv.weight
    # encoder.stages.0.layers.0.norm.weight (scale)
    # encoder.stages.0.layers.0.norm.bias
    
    # Lux keys look like:
    # encoder.stage1.layers.layer_1.conv.weight
    # ...
    
    # Create a clean dictionary for Lux weights
    # Structure: Dict(:layer => Dict(:param => value))
    # We will build it hierarchically or flattened. ComponentArray supports nested NamedTuples.
    
    # Helper to clean keys
    clean_key(k) = replace(k, "module." => "") # If DDP used
    
    lux_ps = Dict{Symbol, Any}()
    
    # Initialize nested dictionaries
    lux_ps[:encoder] = Dict{Symbol, Any}()
    for i in 1:5
        lux_ps[:encoder][Symbol("stage$i")] = Dict{Symbol, Any}()
        # Pre-populate conv/norm dictionaries for layers
        # each stage has 2 layers
        lux_ps[:encoder][Symbol("stage$i")][:layer_1] = Dict{Symbol, Any}(:conv => Dict{Symbol, Any}(), :norm => Dict{Symbol, Any}())
        lux_ps[:encoder][Symbol("stage$i")][:layer_2] = Dict{Symbol, Any}(:conv => Dict{Symbol, Any}(), :norm => Dict{Symbol, Any}())
    end
    
    lux_ps[:decoder] = Dict{Symbol, Any}()
    for i in 1:4
        lux_ps[:decoder][Symbol("block$i")] = Dict{Symbol, Any}(
            :upsample => Dict{Symbol, Any}(),
            :convs => Dict{Symbol, Any}()
        )
        lux_ps[:decoder][Symbol("block$i")][:convs][:layer_1] = Dict{Symbol, Any}(:conv => Dict{Symbol, Any}(), :norm => Dict{Symbol, Any}())
        lux_ps[:decoder][Symbol("block$i")][:convs][:layer_2] = Dict{Symbol, Any}(:conv => Dict{Symbol, Any}(), :norm => Dict{Symbol, Any}())
    end
    
    lux_ps[:final_conv] = Dict{Symbol, Any}()
    
    for (key, tensor) in tensors
        # Skip "all_modules" duplicates
        if contains(key, "all_modules")
            continue
        end
        
        # Process Tensor
        if endswith(key, "conv.weight") || endswith(key, "convs.0.weight") || endswith(key, "transpconvs.0.weight") || contains(key, "transpconvs") && endswith(key, "weight") || contains(key, "seg_layers") && endswith(key, "weight")
             # Conv3D / TransposedConv3D weights need permutedims to fix layout
             # PyTorch: (Out, In, D, H, W) or (In, Out, D, H, W) for Transposed
             # Lux: (W, H, D, In, Out) or (W, H, D, Out, In) for Transposed
             # permutedims(5, 4, 3, 2, 1) handles the dimension mapping correctly.
             # AND we need to flip spatial dimensions (1, 2, 3) because NNlib uses Convolution (x-k) while PyTorch uses Cross-Correlation (x+k).
             
             p = permutedims(tensor, (5, 4, 3, 2, 1))
             val = reverse(p, dims=(1, 2, 3))
        elseif endswith(key, "norm.weight")
             # Rename to scale
             val = tensor
             key = replace(key, "norm.weight" => "norm.scale")
        else
             val = tensor
        end
        
        # --- Mapping Logic ---
        
        # Encoder: `encoder.stages.I.0.convs.J.conv/norm.param`
        m = match(r"encoder\.stages\.(\d+)\.0\.convs\.(\d+)\.(conv|norm)\.(weight|bias|scale)", key)
        if m !== nothing
            stage_idx = parse(Int, m[1]) + 1
            layer_idx = parse(Int, m[2]) + 1
            type = Symbol(m[3])
            param = Symbol(m[4])
            
            # Map param name
            if param == :weight && type == :norm
                param = :scale
            end
            
            lux_ps[:encoder][Symbol("stage$stage_idx")][Symbol("layer_$layer_idx")][type][param] = val
            continue
        end
        
        # Decoder Convs: `decoder.stages.I.convs.J.conv/norm.param`
        m = match(r"decoder\.stages\.(\d+)\.convs\.(\d+)\.(conv|norm)\.(weight|bias|scale)", key)
        if m !== nothing
            block_idx = parse(Int, m[1]) + 1
            layer_idx = parse(Int, m[2]) + 1
            type = Symbol(m[3])
            param = Symbol(m[4])
            
             if param == :weight && type == :norm
                param = :scale
            end
            
            lux_ps[:decoder][Symbol("block$block_idx")][:convs][Symbol("layer_$layer_idx")][type][param] = val
            continue
        end
        
        # Decoder Upsample: `decoder.transpconvs.I.weight/bias`
        m = match(r"decoder\.transpconvs\.(\d+)\.(weight|bias)", key)
        if m !== nothing
            block_idx = parse(Int, m[1]) + 1
            param = Symbol(m[2])
            
            lux_ps[:decoder][Symbol("block$block_idx")][:upsample][param] = val
            continue
        end
        
        # Final Conv: `decoder.seg_layers.3.weight/bias`
        m = match(r"decoder\.seg_layers\.3\.(weight|bias)", key)
        if m !== nothing
            param = Symbol(m[1])
            lux_ps[:final_conv][param] = val
            continue
        end
    end
    
    # Convert to ComponentArray implicitly by structure?
    # Lux models expect a NamedTuple tree.
    # Recursively convert Dict to NamedTuple.
    
    function dict_to_nt(d::Dict)
        return (; (k => (v isa Dict ? dict_to_nt(v) : v) for (k, v) in d)...)
    end
    
    return dict_to_nt(lux_ps)
end


"""
    load_moose_weights(model, safetensors_path)

Load weights for MOOSE (6 Encoder Stages, 5 Decoder Blocks)
"""
function load_moose_weights(model, safetensors_path)
    tensors = SafeTensors.load_safetensors(safetensors_path)
    
    lux_ps = Dict{Symbol, Any}()
    
    # Initialize dictionaries
    lux_ps[:encoder] = Dict{Symbol, Any}()
    for i in 1:6
        lux_ps[:encoder][Symbol("stage$i")] = Dict{Symbol, Any}()
        lux_ps[:encoder][Symbol("stage$i")][:layer_1] = Dict{Symbol, Any}(:conv => Dict{Symbol, Any}(), :norm => Dict{Symbol, Any}())
        lux_ps[:encoder][Symbol("stage$i")][:layer_2] = Dict{Symbol, Any}(:conv => Dict{Symbol, Any}(), :norm => Dict{Symbol, Any}())
    end
    
    lux_ps[:decoder] = Dict{Symbol, Any}()
    for i in 1:5
        lux_ps[:decoder][Symbol("block$i")] = Dict{Symbol, Any}(
            :upsample => Dict{Symbol, Any}(),
            :convs => Dict{Symbol, Any}()
        )
        lux_ps[:decoder][Symbol("block$i")][:convs][:layer_1] = Dict{Symbol, Any}(:conv => Dict{Symbol, Any}(), :norm => Dict{Symbol, Any}())
        lux_ps[:decoder][Symbol("block$i")][:convs][:layer_2] = Dict{Symbol, Any}(:conv => Dict{Symbol, Any}(), :norm => Dict{Symbol, Any}())
    end
    
    lux_ps[:final_conv] = Dict{Symbol, Any}()
    
    for (key, tensor) in tensors
        if contains(key, "all_modules")
            continue
        end
        
        # Process Tensor (Permute)
        if endswith(key, "conv.weight") || endswith(key, "convs.0.weight") || endswith(key, "transpconvs.0.weight") || contains(key, "transpconvs") && endswith(key, "weight") || contains(key, "seg_layers") && endswith(key, "weight")
             p = permutedims(tensor, (5, 4, 3, 2, 1))
             val = reverse(p, dims=(1, 2, 3))
        elseif endswith(key, "norm.weight")
             val = tensor
             key = replace(key, "norm.weight" => "norm.scale")
        else
             val = tensor
        end
        
        # Encoder Mapping
        m = match(r"encoder\.stages\.(\d+)\.0\.convs\.(\d+)\.(conv|norm)\.(weight|bias|scale)", key)
        if m !== nothing
            stage_idx = parse(Int, m[1]) + 1
            layer_idx = parse(Int, m[2]) + 1
            type = Symbol(m[3])
            param = Symbol(m[4])
            
            if param == :weight && type == :norm; param = :scale; end
            
            lux_ps[:encoder][Symbol("stage$stage_idx")][Symbol("layer_$layer_idx")][type][param] = val
            continue
        end
        
        # Decoder Convs
        m = match(r"decoder\.stages\.(\d+)\.convs\.(\d+)\.(conv|norm)\.(weight|bias|scale)", key)
        if m !== nothing
            block_idx = parse(Int, m[1]) + 1
            layer_idx = parse(Int, m[2]) + 1
            type = Symbol(m[3])
            param = Symbol(m[4])
            
            if param == :weight && type == :norm; param = :scale; end
            
            lux_ps[:decoder][Symbol("block$block_idx")][:convs][Symbol("layer_$layer_idx")][type][param] = val
            continue
        end
        
        # Decoder Upsample
        m = match(r"decoder\.transpconvs\.(\d+)\.(weight|bias)", key)
        if m !== nothing
            block_idx = parse(Int, m[1]) + 1
            param = Symbol(m[2])
            lux_ps[:decoder][Symbol("block$block_idx")][:upsample][param] = val
            continue
        end
        
        # Final Conv
        # Check index for final conv. In TotalSeg it was 3?
        # Typically seg_layers has one entry per decoder level if deep supervision.
        # But we only need final output.
        # Check plans/code usually it's last index.
        # MOOSE likely has index 0 or similar if deep supervision matches.
        # Assuming last index corresponds to highest res.
        # In nnU-Net, seg_layers[0] is deepest, seg_layers[-1] is highest res? NO.
        # nnU-Net: seg_layers[0] -> output of bottleneck?
        # Actually usually:
        # decoder.stages[0] -> upsample bottleneck
        # seg_layers[0] -> output of stage 0?
        # Let's inspect weights key names in verification if needed.
        # TotalSegmentator used `decoder.seg_layers.3`. (4 decoder blocks -> index 0,1,2,3).
        # MOOSE has 5 decoder blocks -> likely `decoder.seg_layers.4`.
        
        m = match(r"decoder\.seg_layers\.(\d+)\.(weight|bias)", key)
        if m !== nothing
            idx = parse(Int, m[1])
            param = Symbol(m[2])
            # Assuming 4 is the last one (0-indexed, 5 blocks)
            if idx == 4
                lux_ps[:final_conv][param] = val
            end
            continue
        end
    end
    
    function dict_to_nt(d::Dict)
        return (; (k => (v isa Dict ? dict_to_nt(v) : v) for (k, v) in d)...)
    end
    
    return dict_to_nt(lux_ps)
end

using Lux
using SafeTensors
using ComponentArrays

function load_merlin_weights(model, safetensors_path)
    tensors = SafeTensors.load_safetensors(safetensors_path)
    
    # Initialize model to get structure
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    ps_dict = Dict{Symbol, Any}()
    st_dict = Dict{Symbol, Any}()
    
    # Helper to traverse and set value
    # We will build a ComponentArray-compatible dictionary structure?
    # Or just modify the ComponentArray `ps` directly if possible?
    # ComponentArrays usually immutable in structure, but values mutable? 
    # Better to reconstruct the nested dictionary and convert to ComponentArray/NamedTuple.
    # But `ps` structure is complex.
    # Let's try to mutate the `ps` and `st` structure if they are strictly NamedTuples of Arrays.
    # Actually Lux parameters are NamedTuples. We can't mutate NamedTuples.
    # We must construct a new NamedTuple tree.
    # Or use `ComponentArray(ps)` and then indexing.
    # Let's assume we can map keys to paths.
    
    # Flattening `ps` and `st` is hard.
    # Strategy: recursively walk `ps` and `st`, look for corresponding key in `tensors`.
    
    # Wait, mapping from Flat Keys (PyTorch) to Nested (Lux) is easier with a pattern matcher.
    
    # Create mutable copies of ps and st (as Dicts of Dicts)
    # Then convert back?
    
    function to_nested_dict(x)
        if x isa NamedTuple
            return Dict(k => to_nested_dict(v) for (k, v) in pairs(x))
        else
            return x # Leaf (Array)
        end
    end
    
    ps_d = to_nested_dict(ps)
    st_d = to_nested_dict(st)
    
    println("Initial PS keys: ", keys(ps))
    println("Initial Dict Keys: ", keys(ps_d))
    
    for (key, val) in tensors
        # Key format: "layer1.0.conv1.weight"
        # Lux format: layer1 -> layer_1 -> conv1 -> weight
        
        parts = split(key, ".")
        
        # Traverse
        current_ps = ps_d
        current_st = st_d
        
        # Identify Layer
        # Special handling for "layerX.Y" -> "layerX.layer_{Y+1}"
        
        # We need to construct the path to walk
        path = Symbol[]
        
        i = 1
        while i <= length(parts)
            p = parts[i]
            
            # Check if p is integer (Sequential index)
            if all(isdigit, p)
                idx = parse(Int, p)
                push!(path, Symbol("layer_$(idx + 1)"))
            else
                push!(path, Symbol(p))
            end
            i += 1
        end
        
        # Now walk the path in ps_d or st_d
        # Last part is usually "weight", "bias", "running_mean", "running_var"
        leaf = path[end]
        valid_path = true
        
        pointer = ps_d
        target = :ps
        
        if leaf == :weight || leaf == :bias
            target = :ps
            pointer = ps_d
        elseif leaf == :running_mean || leaf == :running_var
            target = :st
            pointer = st_d
        else
            # num_batches_tracked?
            continue
        end
        
        # Walk
        for p in path[1:end-1]
            if haskey(pointer, p)
                 pointer = pointer[p]
            else
                 # println("Warning: Path not found for $key: missing $p")
                 valid_path = false
                 break
            end
        end
        
        if valid_path
            # Process Value
            param_val = val
            
            # Conv3D Flip Logic
            # "convX.weight" or "downsample.0.weight"
            # If 5D tensor
            if ndims(val) == 5 && (leaf == :weight)
                 # Permute (N, C, D, H, W) -> (W, H, D, C, N) (Layout fix)
                 p = permutedims(val, (5, 4, 3, 2, 1))
                 # Reverse Spatial (1, 2, 3) (Correlation -> Convolution fix)
                 param_val = reverse(p, dims=(1, 2, 3))
                 
                 # Check if size matches
                 # keys in pointer[leaf] should match param_val size?
            elseif leaf == :running_mean || leaf == :running_var
                 # Lux BN params are vectors. PyTorch are vectors.
                 # Just assign.
                 # running_var might need checking if it is std or var. PyTorch is var. Lux is var?
                 # Lux BatchNorm state: (running_mean, running_var)
            elseif leaf == :weight && haskey(pointer, :scale)
                 # BatchNorm weight -> scale
                 leaf = :scale
            end
            
             pointer[leaf] = param_val
        end
    end
    
    # Reconstruct NamedTuples with Layout/Order Preserved
    function restore_structure(template, d)
        if template isa NamedTuple
            # Iterate over template keys to preserve order
            return NamedTuple{keys(template)}(
                restore_structure(getproperty(template, k), d[k]) for k in keys(template)
            )
        elseif d isa Dict
            # Should not happen if d matches template structure, but strictly:
            # If template is not NamedTuple (leaf), return d
            return d
        else
            return d
        end
    end
    
    # We use valid_path logic, but we might have missed some keys if they weren't in safetensors?
    # But we started with ps_d = to_nested_dict(ps), so it has all keys.
    
    new_ps_nt = restore_structure(ps, ps_d)
    new_st = restore_structure(st, st_d)
    
    new_ps = ComponentArray(new_ps_nt)
    
    return new_ps, new_st
end

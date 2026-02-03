using SafeTensors

function inspect_keys(path)
    tensors = SafeTensors.load_safetensors(path)
    keys_list = sort(collect(keys(tensors)))
    
    println("Found $(length(keys_list)) keys.")
    for k in keys_list
        println(k, " -> ", size(tensors[k]))
    end
end

inspect_keys("external_sources/weights/Task297/model_final.safetensors")

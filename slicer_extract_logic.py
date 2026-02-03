import os
import slicer
import sys
import vtk

def sanitize_name(name):
    """Remove illegal filesystem characters."""
    for char in [":", "/", "\\", "*", "?", "\"", "<", ">", "|"]:
        name = name.replace(char, "_")
    return name.strip()

def process_mrb(mrb_path, output_dir):
    print(f"--- Loading Scene: {mrb_path} ---")
    slicer.mrmlScene.Clear(0)
    
    try:
        slicer.util.loadScene(mrb_path)
    except Exception as e:
        print(f"ERROR: Could not load scene: {e}")
        return

    # 1. Process Scalar Volumes (Images)
    nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    for node in nodes:
        # Skip LabelMaps here, handled separately
        if node.IsA("vtkMRMLLabelMapVolumeNode"):
            continue
            
        clean_name = sanitize_name(node.GetName())
        output_path = os.path.join(output_dir, f"{clean_name}.nii.gz")
        
        # Ensure it has a storage node for saving
        if not node.GetStorageNode():
            node.AddDefaultStorageNode()
            
        print(f"  > Saving Volume: {clean_name} -> {output_path}")
        slicer.util.saveNode(node, output_path)

    # 2. Process LabelMap Volumes (Legacy Masks)
    label_nodes = slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")
    for node in label_nodes:
        clean_name = sanitize_name(node.GetName())
        output_path = os.path.join(output_dir, f"{clean_name}_label.nii.gz")
        
        if not node.GetStorageNode():
            node.AddDefaultStorageNode()
            
        print(f"  > Saving LabelMap: {clean_name} -> {output_path}")
        slicer.util.saveNode(node, output_path)

    # 3. Process Segmentation Nodes (Modern Masks)
    seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    seg_logic = slicer.modules.segmentations.logic()
    
    for seg_node in seg_nodes:
        clean_name = sanitize_name(seg_node.GetName())
        print(f"  > Exporting Segmentation: {clean_name}")
        
        # Create a transient LabelMap node
        temp_labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        temp_labelmap.SetName(f"{clean_name}_Export")
        
        # Rasterize all segments into one labelmap
        success = seg_logic.ExportAllSegmentsToLabelmapNode(
            seg_node, 
            temp_labelmap, 
            slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
        )
        
        if success:
            output_path = os.path.join(output_dir, f"{clean_name}_mask.nii.gz")
            if not temp_labelmap.GetStorageNode():
                temp_labelmap.AddDefaultStorageNode()
            slicer.util.saveNode(temp_labelmap, output_path)
            print(f"    - Saved to {output_path}")
        else:
            print(f"    - FAILED to export segmentation {clean_name}")
            
        slicer.mrmlScene.RemoveNode(temp_labelmap)

    print(f"--- Finished Case: {os.path.basename(mrb_path)} ---")

if __name__ == "__main__":
    # Get paths from environment variables
    mrb = os.environ.get("MRB_INPUT")
    out = os.environ.get("MRB_OUTPUT")
    
    if mrb and out:
        process_mrb(mrb, out)
    else:
        print("ERROR: MRB_INPUT or MRB_OUTPUT environment variables not set.")
    
    sys.exit(0)

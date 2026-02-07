import xnat
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

server_url = "https://imaging-platform.diz-ag.med.ovgu.de"
username = "1823514a-11f5-48e8-a56b-e2d8f0fe546d" 
password = "2oxa2hiv8YXXuOQIkFNCk449CtXOrwkEiTmVUSFZdey1FMF68EeJ8zuT5COLrArB"
project_id = "Prostata_bimodal_PIPELINE"

print(f"Connecting to {server_url}...")
with xnat.connect(server_url, user=username, password=password, verify=False) as session:
    print(f"Successfully connected.")
    project = session.projects[project_id]
    print(f"Found project: {project.name}")
    
    # Get only the first subject for exploration
    subject = list(project.subjects.values())[0]
    print(f"\n=== Exploring Subject: {subject.label} ===")
    
    print(f"\n--- Subject-Level Resources ---")
    for res_name, res in subject.resources.items():
        print(f"  Resource: {res_name} (label: {res.label})")
        for f_name, f in list(res.files.items())[:5]:
            print(f"    File: {f.name}")
        if len(res.files) > 5:
            print(f"    ... and {len(res.files) - 5} more files")
    
    print(f"\n--- Subject Experiments ---")
    for exp_name, exp in subject.experiments.items():
        print(f"\n  Experiment: {exp.label}")
        
        print(f"    Experiment-Level Resources:")
        for res_name, res in exp.resources.items():
            print(f"      Resource: {res_name} (label: {res.label})")
            for f_name, f in list(res.files.items())[:3]:
                print(f"        File: {f.name}")
            if len(res.files) > 3:
                print(f"        ... and {len(res.files) - 3} more files")
        
        # Try to access scans if available (common in imaging sessions)
        if hasattr(exp, 'scans'):
            print(f"    Scans:")
            for scan_id, scan in list(exp.scans.items())[:5]:
                print(f"      Scan: {scan_id} (type: {getattr(scan, 'type', 'N/A')})")
                if hasattr(scan, 'resources'):
                    for res_name, res in scan.resources.items():
                        print(f"        Scan Resource: {res_name} (label: {res.label})")
                        for f_name, f in list(res.files.items())[:2]:
                            print(f"          File: {f.name}")
            if len(exp.scans) > 5:
                print(f"      ... and {len(exp.scans) - 5} more scans")

    # Search deep for verification_scene.mrb
    print(f"\n\n--- Deep search for 'verification_scene.mrb' or 'slicer' ---")
    
    def search_resources(resources, prefix=""):
        found_items = []
        for res_name, res in resources.items():
            res_label_lower = res.label.lower() if res.label else ""
            if "slicer" in res_label_lower:
                print(f"  {prefix}Found 'slicer' resource: {res.label}")
            for f_name, f in res.files.items():
                if "verification_scene" in f.name.lower() or "slicer" in res_label_lower:
                    found_items.append((res.label, f.name, prefix))
                    print(f"  {prefix}Match in resource '{res.label}': {f.name}")
        return found_items
    
    all_found = []
    all_found.extend(search_resources(subject.resources, "Subject > "))
    
    for exp_name, exp in subject.experiments.items():
        all_found.extend(search_resources(exp.resources, f"Experiment '{exp.label}' > "))
        if hasattr(exp, 'scans'):
            for scan_id, scan in exp.scans.items():
                if hasattr(scan, 'resources'):
                    all_found.extend(search_resources(scan.resources, f"Experiment '{exp.label}' > Scan {scan_id} > "))

    if not all_found:
        print("  No 'verification_scene.mrb' or 'slicer' resources found for this subject.")

print("Exploration complete.")

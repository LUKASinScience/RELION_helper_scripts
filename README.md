# RELION_helper_scripts
## RELION Helper Scripts provide helpful Python tools for cryo-EM workflow:  

### Workflow Documentation: Creates visual RELION job trees (PNG/SVG) documenting job parameters and pipeline history.  

The make_relion_job_tree_png.py script is designed for complete workflow documentation. It generates a visual representation of the entire upstream pipeline for a single selected RELION job, including all preceding jobs, their settings, and input data.

    Function: Generates a detailed job tree as a PNG (or optional SVG), presenting all job parameters in readable cards, organized by RELION GUI tabs.

    Placement: The script must be placed and run from within the specific RELION processing directory.

    Output Location: The results are saved in a new subdirectory named after the current processing directory, with the suffix _jobtrees appended.

        Example: If run inside /data/relion_project/, the output folder will be /data/relion_project/relion_project_jobtrees/.

### STAR File Modification/Curation: Scripts for modifying, validating, and ensuring consistency across various .star files (e.g., managing optics groups)

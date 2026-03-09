import pathlib

def display_tree(directory, prefix=""):
    """
    Recursively prints a visual tree of the given directory.
    """
    path = pathlib.Path(directory)
    
    # Get a sorted list of files and folders, ignoring hidden ones
    paths = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
    
    for i, p in enumerate(paths):
        # Check if this is the last item in the current folder
        is_last = (i == len(paths) - 1)
        
        # Select the appropriate branch character
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{p.name}")
        
        # If it's a directory, recurse into it
        if p.is_dir():
            # Adjust the prefix for the next level
            extension = "    " if is_last else "│   "
            display_tree(p, prefix + extension)

if __name__ == "__main__":
    # Change '.' to any specific path you want to inspect
    root_dir = "./fu"
    print(f"Project Tree: {pathlib.Path(root_dir).absolute().name}")
    display_tree(root_dir)
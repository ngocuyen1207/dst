import os

def is_relevant(name):
    ignored = {"__pycache__", ".ipynb_checkpoints", ".git", ".DS_Store", "venv", "env", "node_modules"}
    if name in ignored or name.startswith("."):
        return False
    return True

def print_tree(root, indent=""):
    try:
        files = sorted([f for f in os.listdir(root) if is_relevant(f)])
    except PermissionError:
        return
    for i, name in enumerate(files):
        path = os.path.join(root, name)
        is_last = i == len(files) - 1
        print(indent + ("└── " if is_last else "├── ") + name)
        if os.path.isdir(path):
            print_tree(path, indent + ("    " if is_last else "│   "))

if __name__ == "__main__":
    print_tree(".")

"""
Script to fix mlflow circular import issue
This patches mlflow's __init__.py to avoid circular imports
"""
import os
import sys
import shutil

def fix_mlflow_import():
    """Fix mlflow circular import by patching __init__.py"""
    try:
        import mlflow
        mlflow_path = os.path.dirname(mlflow.__file__)
    except:
        # Try to find mlflow manually
        for path in sys.path:
            mlflow_path = os.path.join(path, 'mlflow')
            if os.path.exists(mlflow_path) and os.path.isdir(mlflow_path):
                break
        else:
            print("ERROR: Could not find mlflow installation")
            return False
    
    init_file = os.path.join(mlflow_path, '__init__.py')
    backup_file = os.path.join(mlflow_path, '__init__.py.backup')
    
    if not os.path.exists(init_file):
        print(f"ERROR: Could not find {init_file}")
        return False
    
    # Read the current __init__.py
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if '# MLFLOW CIRCULAR IMPORT FIX' in content:
        print("mlflow is already patched")
        return True
    
    # Create backup
    shutil.copy2(init_file, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Patch the file - wrap the version import in try-except
    lines = content.split('\n')
    patched_lines = []
    patched = False
    
    for i, line in enumerate(lines):
        # Look for the version import line
        if 'from mlflow.version import VERSION' in line or 'from mlflow.version import' in line:
            # Replace with try-except version
            patched_lines.append('# MLFLOW CIRCULAR IMPORT FIX')
            patched_lines.append('try:')
            patched_lines.append('    from mlflow.version import VERSION as __version__')
            patched_lines.append('except (ImportError, AttributeError):')
            patched_lines.append('    # Fallback if version import fails')
            patched_lines.append('    try:')
            patched_lines.append('        import mlflow.version as _version_module')
            patched_lines.append('        __version__ = getattr(_version_module, "VERSION", "2.5.0")')
            patched_lines.append('    except:')
            patched_lines.append('        __version__ = "2.5.0"')
            patched = True
        else:
            patched_lines.append(line)
    
    if not patched:
        print("WARNING: Could not find version import line to patch")
        print("Trying alternative patching method...")
        # Try to add the fix at the beginning
        new_content = '# MLFLOW CIRCULAR IMPORT FIX\n'
        new_content += 'try:\n'
        new_content += '    from mlflow.version import VERSION as __version__\n'
        new_content += 'except (ImportError, AttributeError):\n'
        new_content += '    try:\n'
        new_content += '        import mlflow.version as _version_module\n'
        new_content += '        __version__ = getattr(_version_module, "VERSION", "2.5.0")\n'
        new_content += '    except:\n'
        new_content += '        __version__ = "2.5.0"\n\n'
        new_content += content
        
        # Remove the original version import if it exists
        if 'from mlflow.version import VERSION as __version__' in new_content:
            new_content = new_content.replace('from mlflow.version import VERSION as __version__', '# Removed duplicate import')
        
        patched_lines = new_content.split('\n')
        patched = True
    
    # Write patched file
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(patched_lines))
    
    print(f"Successfully patched {init_file}")
    return True

if __name__ == '__main__':
    print("Fixing mlflow circular import issue...")
    if fix_mlflow_import():
        print("SUCCESS: mlflow has been patched")
        print("Try importing mlflow now:")
        try:
            import mlflow
            print(f"mlflow imported successfully! Version: {mlflow.__version__}")
        except Exception as e:
            print(f"ERROR: Still cannot import mlflow: {e}")
            sys.exit(1)
    else:
        print("FAILED: Could not patch mlflow")
        sys.exit(1)


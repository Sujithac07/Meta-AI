#!/usr/bin/env python3
import os
import shutil
import sys

def main():
    # Directory path
    dir_path = r"C:\Users\sujit\OneDrive\Desktop\meta_ai_builder_pp\frontend\react-dashboard"
    index_file = os.path.join(dir_path, "index.html")
    new_file = os.path.join(dir_path, "dashboard_new.html")
    
    print(f"Working directory: {dir_path}")
    print(f"Index file: {index_file}")
    print(f"New file: {new_file}")
    print()
    
    # Delete old index.html
    if os.path.exists(index_file):
        try:
            os.remove(index_file)
            print("✓ Deleted old index.html")
        except Exception as e:
            print(f"✗ Error deleting index.html: {e}")
            return 1
    else:
        print("⚠ Old index.html not found")
    
    # Copy dashboard_new.html to index.html
    if os.path.exists(new_file):
        try:
            shutil.copy2(new_file, index_file)
            print("✓ Copied dashboard_new.html to index.html")
        except Exception as e:
            print(f"✗ Error copying file: {e}")
            return 1
    else:
        print("✗ dashboard_new.html not found")
        return 1
    
    # Verify new index.html
    if os.path.exists(index_file):
        try:
            size = os.path.getsize(index_file)
            print(f"✓ New index.html exists ({size} bytes)")
            
            # Show first few lines
            with open(index_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]
                print(f"\nFirst {len(lines)} lines of new index.html:")
                for i, line in enumerate(lines, 1):
                    print(f"  {i}. {line.rstrip()}")
            return 0
        except Exception as e:
            print(f"✗ Error reading index.html: {e}")
            return 1
    else:
        print("✗ index.html verification failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

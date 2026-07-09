import urllib.request
import bz2
import os

url = "http://ciscobinary.openh264.org/openh264-2.5.0-win64.dll.bz2"
bz2_path = "openh264-2.5.0-win64.dll.bz2"
dll_path = "openh264-2.5.0-win64.dll"

print("Downloading OpenH264 library from Cisco...")
try:
    urllib.request.urlretrieve(url, bz2_path)
    print("Downloaded. Decompressing...")
    
    with bz2.open(bz2_path, "rb") as source, open(dll_path, "wb") as dest:
        dest.write(source.read())
        
    print(f"Success: Extracted {dll_path} to current directory.")
    
    # Also copy to venv/Scripts just in case
    venv_scripts = os.path.join("venv", "Scripts")
    if os.path.exists(venv_scripts):
        import shutil
        shutil.copy(dll_path, os.path.join(venv_scripts, dll_path))
        print(f"Success: Copied {dll_path} to {venv_scripts}.")
        
    # Clean up bz2 file
    os.remove(bz2_path)
except Exception as e:
    print(f"Error occurred: {e}")

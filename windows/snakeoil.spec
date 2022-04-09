# -*- mode: python ; coding: utf-8 -*-
from kivy_deps import sdl2, glew

ROOT_DIR = "C:\\src\\snake-oil"

block_cipher = None


a = Analysis([f"{ROOT_DIR}\\main.py"],
             pathex=[],
             binaries=[],
             datas=[
                 (f"{ROOT_DIR}\\snakeoil.kv", "."),
                 (f"{ROOT_DIR}\\audio\\*.flac", "audio"),
                 (f"{ROOT_DIR}\\venv\\Lib\\site-packages\\librosa\\util\\example_data", "librosa\\util\\example_data")
             ],
             hiddenimports=["sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes"],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='snakeoil',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               upx_exclude=[],
               name='snakeoil')

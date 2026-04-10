import pkgutil

import pysstv

print("Модули pysstv:")
for importer, modname, ispkg in pkgutil.walk_packages(pysstv.__path__):
    print(f"  {modname} (пакет: {ispkg})")

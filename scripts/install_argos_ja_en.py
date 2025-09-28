from argostranslate import package
pkgs = package.get_available_packages()
pkg = next(p for p in pkgs if p.from_code=="ja" and p.to_code=="en")
package.install_from_path(pkg.download())
print("Installed:", pkg)

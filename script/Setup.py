import sys
import subprocess
import importlib.util as importlib_util

cudaPackage = {"torch": 'i', "torchvision": 'i'}
dxPackage = {"torchvision": 'i', "torch": 'u', "torch-directml": 'i'}

class PythonConfiguration:
    @classmethod
    def Validate(cls):
        if not cls.__ValidatePython():
            return # cannot validate further
        
        reply = str(input("Use CUDA_12.4 or Dx? [1/2]: ")).lower().strip()[:1]
        if reply == '1':
            for package in cudaPackage.items():
                cls.__InstallPackage(package, url = 'https://download.pytorch.org/whl/cu124')
        elif reply == '2':
            for package in dxPackage.items():
                cls.__InstallPackage(package)

    @classmethod
    def __ValidatePython(cls, versionMajor = 3, versionMinor = 3):
        if sys.version is not None:
            print("Python version {0:d}.{1:d}.{2:d} detected.".format( \
                sys.version_info.major, sys.version_info.minor, sys.version_info.micro))
            if sys.version_info.major < versionMajor or (sys.version_info.major == versionMajor and sys.version_info.minor < versionMinor):
                print("Python version too low, expected version {0:d}.{1:d} or higher.".format( \
                    versionMajor, versionMinor))
                return False
            return True

    @classmethod
    def __ValidatePackage(cls, package):
        [packageName, tag] = package
        if importlib_util.find_spec(packageName) is None:
            return cls.__InstallPackage(package)
        return True

    @classmethod
    def __InstallPackage(cls, package, url = 'https://pypi.tuna.tsinghua.edu.cn/simple'):
        [packageName, tag] = package
        if tag == 'i':
            print(f"Installing {packageName} module...")
            subprocess.check_call(['python', '-m', 'pip', 'install', packageName, '-i', url])
            return cls.__ValidatePackage(package)
        elif tag == 'u':
            print(f"Uninstalling {packageName} module...")
            subprocess.check_call(['python', '-m', 'pip3', 'uninstall', '-y', packageName])
        return True

if __name__ == "__main__":
    PythonConfiguration.Validate()
import subprocess
import sys
from typing import List

def get_requirements() -> List[str]:
    """Read and parse requirements.txt file."""
    try:
        with open('requirements.txt', 'r') as file:
            return [line.strip() for line in file if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("Error: requirements.txt not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading requirements.txt: {e}")
        sys.exit(1)

def uninstall_packages(packages: List[str]) -> None:
    """Uninstall the specified packages."""
    print("Uninstalling existing packages...")
    for package in packages:
        package_name = package.split('==')[0].split('>=')[0]  # Handle both == and >= version specifications
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name])
            print(f"Successfully uninstalled {package_name}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Error uninstalling {package_name}: {e}")
            # Continue with other packages even if one fails
            continue
        except Exception as e:
            print(f"Warning: Unexpected error uninstalling {package_name}: {e}")
            continue

def install_requirements() -> None:
    """Install packages from requirements.txt."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    print("Starting package management process...")
    requirements = get_requirements()
    uninstall_packages(requirements)
    install_requirements()
    print("\nPackage management process completed!")

if __name__ == "__main__":
    main()
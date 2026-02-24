import platform
import sys


def main():
    print("=== Simple Training Script ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print("Pretend we are training a model here...")
    # Later, this is where MLflow logging would go.


if __name__ == "__main__":
    main()

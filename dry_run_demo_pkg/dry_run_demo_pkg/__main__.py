"""
Main entry point for dry_run_demo_pkg.
"""

from . import hello

def main():
    """Main function demonstrating package usage."""
    print("Dry-run demo package scaffold launched.")
    print(hello())
    print("\nThis is a minimal scaffold for Vetinari end-to-end dry-run demonstration.")
    print("No runtime side effects - this is a demonstration only.")

if __name__ == "__main__":
    main()

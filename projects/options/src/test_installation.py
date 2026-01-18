#!/usr/bin/env python3
"""Quick test script to verify the installation works."""

import sys

def test_imports():
    """Test that all imports work."""
    print("=" * 60)
    print("Testing Package Installation")
    print("=" * 60)

    try:
        from options_analysis import AnalysisToolkit
        print("✓ AnalysisToolkit imported")
    except ImportError as e:
        print(f"✗ Failed to import AnalysisToolkit: {e}")
        return False

    try:
        from options_analysis import OptionsAnalysisOrchestrator
        print("✓ OptionsAnalysisOrchestrator imported")
    except ImportError as e:
        print(f"✗ Failed to import OptionsAnalysisOrchestrator: {e}")
        return False

    try:
        from options_analysis.cli import main as cli_main
        print("✓ CLI module imported")
    except ImportError as e:
        print(f"✗ Failed to import CLI: {e}")
        return False

    return True


def test_toolkit():
    """Test basic toolkit functionality."""
    print("\n" + "=" * 60)
    print("Testing Toolkit Functionality")
    print("=" * 60)

    try:
        from options_analysis import AnalysisToolkit

        toolkit = AnalysisToolkit()
        print("✓ Toolkit initialized")

        # Test get_quote (quick API call)
        print("\nTesting get_quote('AAPL')...")
        quote = toolkit.get_quote('AAPL')

        if quote and 'price' in quote:
            print(f"✓ Quote retrieved successfully")
            print(f"  AAPL Price: ${quote['price']:.2f}")
            print(f"  Timestamp: {quote['timestamp']}")
            return True
        else:
            print("✗ Quote data incomplete")
            return False

    except Exception as e:
        print(f"✗ Error testing toolkit: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🧪 Options Analysis Toolkit - Installation Test\n")

    # Test imports
    imports_ok = test_imports()

    if not imports_ok:
        print("\n❌ Import test failed!")
        print("Try reinstalling: uv pip install -e '.[all]'")
        sys.exit(1)

    # Test toolkit
    toolkit_ok = test_toolkit()

    print("\n" + "=" * 60)
    if imports_ok and toolkit_ok:
        print("✅ All tests passed!")
        print("\nYou're ready to go! Try these commands:")
        print("  1. CLI: .venv/bin/options-elasticity AAPL --top 5")
        print("  2. Python: python examples/toolkit_demo.py")
        print("  3. Legacy: python src/main.py --ticker AAPL")
    elif imports_ok:
        print("⚠️  Imports OK but API test failed")
        print("This might be a network/API issue. Core functionality is installed.")
    else:
        print("❌ Tests failed!")
        print("Try reinstalling the package.")
    print("=" * 60)


if __name__ == '__main__':
    main()

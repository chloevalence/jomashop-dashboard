"""
Test script for PDF parser.
Use this to test the PDF parser with your PDF files and see what data is extracted.
"""
import sys
from pathlib import Path
from pdf_parser import parse_pdf_from_bytes

def test_pdf_parser(pdf_path: str):
    """Test the PDF parser with a local PDF file."""
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"📄 Testing PDF parser with: {pdf_file.name}")
    print("=" * 60)
    
    # Read PDF file
    with open(pdf_file, 'rb') as f:
        pdf_bytes = f.read()
    
    # Parse PDF
    result = parse_pdf_from_bytes(pdf_bytes, pdf_file.name)
    
    if result:
        print("✅ PDF parsed successfully!")
        print("\nExtracted data:")
        print("-" * 60)
        for key, value in result.items():
            print(f"{key:25s}: {value}")
    else:
        print("❌ Failed to parse PDF")
    
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_parser.py <path_to_pdf>")
        print("\nExample:")
        print("  python test_pdf_parser.py '20250715_060053_TYIO-bpagent030844482%40nextiva.com-%2B14178141239-IN.pdf'")
        sys.exit(1)
    
    test_pdf_parser(sys.argv[1])


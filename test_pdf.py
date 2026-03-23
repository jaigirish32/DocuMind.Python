import pdfplumber

PDF_PATH = r"C:\mywork\temp\ODC_MIDAS Accident Users Manual-FOR DEVELOPMENT.pdf"

with pdfplumber.open(PDF_PATH) as pdf:

    print(f"Total pages: {len(pdf.pages)}")
    print()

    # Test page 6 - two column layout
    page = pdf.pages[5]
    print("=== PAGE 6 ===")

    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        use_text_flow=True,
    )
    print(f"Words found: {len(words)}")
    print()

    # Group into lines by y position
    lines = {}
    for w in words:
        y = round(w["top"] / 5) * 5
        if y not in lines:
            lines[y] = []
        lines[y].append(w["text"])

    print("First 15 lines:")
    for y in sorted(lines.keys())[:15]:
        print(" ".join(lines[y]))
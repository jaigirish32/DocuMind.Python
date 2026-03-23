from DocuMind.Documents.Models.Document import (
    ElementType, BoundingBox, Element, Page, Document
)

# Test ElementType
print("ElementType:", ElementType.PARAGRAPH)
print("ElementType value:", ElementType.PARAGRAPH.value)

# Test BoundingBox
box = BoundingBox(x0=10.0, top=20.0, x1=200.0, bottom=32.0)
print("BoundingBox:", box.x0, box.top, box.x1, box.bottom)

# Test Element
elem = Element(
    element_type=ElementType.PARAGRAPH,
    bounds=box,
    text="Total revenues were $96,773M in 2023."
)
print("Element type:", elem.element_type)
print("Element text:", elem.text)

# Test Page
page = Page(page_number=45)
page.elements.append(elem)
page.elements.append(Element(
    element_type=ElementType.TABLE,
    text="Revenue | $96,773M"
))
print("Page:", page.page_number, "elements:", len(page.elements))

# Test Document
doc = Document(source_path="tesla.pdf")
doc.pages.append(page)
print("Document:", doc.source_path, "pages:", len(doc.pages))

print()
print("All Document models OK")
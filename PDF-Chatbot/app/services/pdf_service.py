import fitz


async def extract_text_from_pdf(pdf_file):
    pdf_bytes = await pdf_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text



def extract_pages_from_pdf(pdf_path: str):

    pages = []

    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    doc.close()

    return pages
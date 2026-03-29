import unicodedata
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bookworm.types import Document, Source


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """ """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if not chunk_size > chunk_overlap:
            raise ValueError(
                "Can't have an overlap larger than the chunk size. Got "
                f"`chunk_size` of {chunk_size} and `chunk_overlap` of "
                f"{chunk_overlap}"
            )
        if chunk_overlap > 0.2 * chunk_size:
            logger.warning(
                "Larger chunk overlaps result in more computational cost, "
                "though they can be helpful with continuity in context. Make "
                "sure to pick parameters that work for your use case."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def process(self, document: Document) -> list[Document]:
        chunks = self.splitter.split_text(document.content)
        child_docs = []
        for i, chunk in enumerate(chunks):
            chunk_name = f"{document.name}-chunk-{i}"
            norm_chunk = unicodedata.normalize("NFC", chunk)
            doc = Document(
                parent_id=document.id,
                name=chunk_name,
                source=Source(),
                content=norm_chunk,
            )
            child_docs.append(doc)

        return child_docs

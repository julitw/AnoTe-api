from sqlalchemy import Column, Integer, String, LargeBinary
from .database import Base

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file_data = Column(LargeBinary, nullable=False)  # Przechowywanie pliku w bazie
    file_name = Column(String, nullable=False)  # Oryginalna nazwa pliku
    column_text_name = Column(String)
    column_label_name = Column(String)
    available_labels = Column(String)
    last_annotated_index = Column(Integer, default=0)
    row_count = Column(Integer, default=0)

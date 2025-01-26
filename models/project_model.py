from sqlalchemy import Column, Integer, String
from .database import Base

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file_path = Column(String)
    column_text_name = Column(String)
    column_label_name = Column(String)
    available_labels = Column(String)
    last_annotated_index = Column(Integer, default=0)

from sqlalchemy import Column, Integer, String, LargeBinary
from .database import Base

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file_data = Column(LargeBinary, nullable=False)  
    file_name = Column(String, nullable=False)  
    column_text_name = Column(String)
    column_label_name = Column(String)
    available_labels = Column(String)
    row_count = Column(Integer, default=0)
    modified_file_data = Column(LargeBinary, nullable=True)
    number_evaluated_data = Column(Integer, nullable=True)
    number_positive_evaluated_data = Column(Integer, nullable=True)
    number_annotated_data = Column(Integer, nullable=True)

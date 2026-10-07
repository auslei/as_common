from pydantic import BaseModel, Field
from typing import Optional
import json
import os

class ProjectMetadata(BaseModel):
    project_name: str = Field(..., description="The name of the project")
    language: str = Field("en", description="Language code for the project (e.g. en, zh)")
    sector: str = Field(..., description="The industry sector of the project")
    description: str = Field("", description="A brief description of the project")

    def save_to_folder(self, project_path: str):
        """Save metadata to project.json in the specified folder."""
        file_path = os.path.join(project_path, "project.json")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_folder(cls, project_path: str) -> "ProjectMetadata":
        """Load metadata from project.json in the specified folder."""
        file_path = os.path.join(project_path, "project.json")
        if not os.path.exists(file_path):
            folder_name = os.path.basename(project_path.rstrip("/\\"))
            return cls(project_name=folder_name, sector="General")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return cls(**data)

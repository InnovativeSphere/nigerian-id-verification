"""
config.example.py — Template configuration for the Nigerian ID Verification System.
Rename this file to config.py and fill in your actual values,
or (preferred) create a .env file and let config.py load from there.
"""

from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="id_verification")
    db_user: str = Field(default="postgres")
    db_password: str = Field(default="")   # fill via .env
    camera_index: int = Field(default=0, ge=0)
    cascade_path: str = Field(default="cascades/haarcascade_frontalface_default.xml")
    face_output_dir: str = Field(default="extracted_faces")
    log_file: str = Field(default="logs/verification.log")
    frame_stability_count: int = Field(default=10, ge=1)

    @field_validator('db_password')
    @classmethod
    def password_not_empty(cls, v):
        if not v:
            raise ValueError("DB_PASSWORD must be set in .env")
        return v

    @field_validator('cascade_path', 'face_output_dir', 'log_file')
    @classmethod
    def no_trailing_slash(cls, v):
        return v.rstrip('/\\')


settings = Settings()
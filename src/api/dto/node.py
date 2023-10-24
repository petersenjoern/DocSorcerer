"""DTOs related to Nodes"""

from typing import Optional
from pydantic import BaseModel


class NodeWithEvidence(BaseModel):
    """Node information in the service layer"""
    node_id: str
    text: str
    score: Optional[float]
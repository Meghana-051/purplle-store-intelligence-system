# schema/events_schema.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class CustomerLog(BaseModel):
    customer_id: int
    entry_timestamp: str
    exit_timestamp: Optional[str] = None
    dwell_time_seconds: float
    status: str

class StoreAnalyticsResponse(BaseModel):
    store_id: str
    timestamp: datetime
    total_unique_customers: int
    average_dwell_time_seconds: float
    active_occupancy: int
    people_metrics: List[CustomerLog]
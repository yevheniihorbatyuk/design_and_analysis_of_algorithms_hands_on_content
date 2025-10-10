from dataclasses import dataclass

@dataclass
class UsageEvent:
    user_id:str  
    music_id:str 
    timestamp:str 
    duration_point:str 
    completed:str 
    device:str 
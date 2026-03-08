"""
Temporal models for 4-dimensional time tracking in FinancialKG
"""
from typing import Optional, Union
from datetime import datetime, date, time
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class MarketSession(str, Enum):
    """Market trading session types"""
    PRE_MARKET = "pre_market"      # Before 9:15 AM IST
    REGULAR = "regular"            # 9:15 AM - 3:30 PM IST
    POST_MARKET = "post_market"    # After 3:30 PM IST
    CLOSED = "closed"              # Market closed (weekend/holiday)


class TimeRange(BaseModel):
    """Time range for events/impacts"""
    start: datetime = Field(description="Start time")
    end: Optional[datetime] = Field(default=None, description="End time (None = unknown/ongoing)")
    
    def __str__(self):
        if self.end:
            return f"[{self.start.isoformat()} to {self.end.isoformat()}]"
        return f"[{self.start.isoformat()} to .]"
    
    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds (if end is known)"""
        if self.end:
            return (self.end - self.start).total_seconds()
        return None


class TemporalAttributes(BaseModel):
    """
    4-Dimensional temporal model for financial relationships
    
    Unlike ATOM's 2-time model (t_obs, t_valid), we track 4 temporal dimensions:
    - t_announce: When information was announced/published
    - t_effective: When the information takes legal/business effect
    - t_observe: When we collected/processed the data
    - t_impact: Time window when price/market impact occurred
    """
    
    # Core temporal dimensions
    t_announce: Optional[datetime] = Field(
        default=None,
        description="When information was announced (e.g., news timestamp, press release)"
    )
    
    t_effective: Optional[datetime] = Field(
        default=None,
        description="When information takes effect (e.g., policy start date, merger completion)"
    )
    
    t_observe: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="When we collected/processed this data"
    )
    
    t_impact_start: Optional[datetime] = Field(
        default=None,
        description="Start of price/market impact window"
    )
    
    t_impact_end: Optional[datetime] = Field(
        default=None,
        description="End of price/market impact window"
    )
    
    # Market context
    market_session: Optional[MarketSession] = Field(
        default=None,
        description="Market session when announced/observed"
    )
    
    trading_day: Optional[date] = Field(
        default=None,
        description="Trading day (adjusted for weekends/holidays)"
    )
    
    # Observation history (for temporal tracking like ATOM)
    observation_dates: list[datetime] = Field(
        default_factory=list,
        description="All dates when this relationship was observed (for tracking updates)"
    )
    
    @field_validator('t_announce', 't_effective', 't_observe', 't_impact_start', 't_impact_end', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from string or return None for unparseable values."""
        import re as _re
        if v is None or v == "":
            return None
        if isinstance(v, (datetime, date)):
            return v
        if not isinstance(v, str):
            return None
        v = v.strip()

        # --- explicit skip-list (non-date strings Gemini commonly returns) ---
        _SKIP = {
            "YYYY-MM-DD", "UNKNOWN", "N/A", "NA", "NULL", "NONE",
            "FUTURE", "ONGOING", "IMMEDIATE", "CURRENT", "HISTORICAL",
            "P_CURRENT", "P_FUTURE", "P_PAST",
        }
        if v.upper() in _SKIP:
            return None

        # reject anything that is clearly not date-like (no digit at all)
        if not _re.search(r'\d', v):
            return None

        # --- range strings like "2025-04-01 to 2026-03-31"
        #     "2025-04-01/2025-06-30"  → take the start date
        range_match = _re.match(
            r'(\d{4}-\d{2}-\d{2})\s*(?:to|/)\s*\d{4}-\d{2}-\d{2}', v
        )
        if range_match:
            v = range_match.group(1)

        # --- "prior to YYYY-MM-DD", "after YYYY-MM-DD", "before YYYY-MM-DD"
        rel_match = _re.search(r'(\d{4}-\d{2}-\d{2})', v)

        # --- year-only: "2024", "2025"
        if _re.fullmatch(r'\d{4}', v):
            return datetime(int(v), 1, 1)

        # --- year-month: "2025-07", "2025-05"
        ym = _re.fullmatch(r'(\d{4})-(\d{2})', v)
        if ym:
            return datetime(int(ym.group(1)), int(ym.group(2)), 1)

        # --- standard formats
        for fmt in (
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%d-%m-%Y',
            '%d/%m/%Y',
        ):
            try:
                return datetime.strptime(v, fmt)
            except ValueError:
                continue

        # --- try extracting any YYYY-MM-DD substring as fallback
        if rel_match:
            try:
                return datetime.strptime(rel_match.group(1), '%Y-%m-%d')
            except ValueError:
                pass

        # --- anything else (fuzzy text like "near term", "FY25-27") → None
        return None
    
    @property
    def impact_range(self) -> Optional[TimeRange]:
        """Get impact time range"""
        if self.t_impact_start:
            return TimeRange(start=self.t_impact_start, end=self.t_impact_end)
        return None
    
    def add_observation(self, obs_date: datetime) -> None:
        """Add an observation timestamp"""
        if obs_date not in self.observation_dates:
            self.observation_dates.append(obs_date)
            self.observation_dates.sort()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }


def get_market_session(dt: datetime) -> MarketSession:
    """
    Determine market session based on time (IST)
    
    Args:
        dt: Datetime object (should be in IST)
        
    Returns:
        MarketSession enum
    """
    # Handle weekends
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return MarketSession.CLOSED
    
    # Check time (assuming IST)
    t = dt.time()
    
    # Market timings (IST):
    # Pre-market: Before 9:15 AM
    # Regular: 9:15 AM - 3:30 PM
    # Post-market: After 3:30 PM
    
    pre_market_end = time(9, 15)
    regular_end = time(15, 30)
    
    if t < pre_market_end:
        return MarketSession.PRE_MARKET
    elif t < regular_end:
        return MarketSession.REGULAR
    else:
        return MarketSession.POST_MARKET


def get_trading_day(dt: datetime) -> date:
    """
    Get the trading day for a given datetime
    (adjusts for weekends - maps to next Monday)
    
    Args:
        dt: Datetime object
        
    Returns:
        Trading day as date
    """
    d = dt.date()
    
    # If Saturday, move to next Monday
    if dt.weekday() == 5:
        from datetime import timedelta
        d = d + timedelta(days=2)
    # If Sunday, move to next Monday
    elif dt.weekday() == 6:
        from datetime import timedelta
        d = d + timedelta(days=1)
    
    return d


if __name__ == "__main__":
    # Test temporal models
    from datetime import timedelta
    
    # Example: RBI interest rate hike announcement
    now = datetime.now()
    
    temporal = TemporalAttributes(
        t_announce=now - timedelta(days=1),  # Announced yesterday
        t_effective=now + timedelta(days=7),  # Effective in 7 days
        t_observe=now,                        # Observed now
        t_impact_start=now - timedelta(hours=1),  # Immediate market impact
        t_impact_end=now + timedelta(days=2),     # Impact window 2 days
        market_session=get_market_session(now),
        trading_day=get_trading_day(now)
    )
    
    print("Temporal Attributes:")
    print(f"  Announced: {temporal.t_announce}")
    print(f"  Effective: {temporal.t_effective}")
    print(f"  Observed: {temporal.t_observe}")
    print(f"  Impact: {temporal.impact_range}")
    print(f"  Session: {temporal.market_session}")
    print(f"  Trading Day: {temporal.trading_day}")

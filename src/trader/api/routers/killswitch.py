"""Kill switch control endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from trader.api.schemas.common import KillSwitchResponse
from trader.services.system_state import SystemStateStore

router = APIRouter(prefix="/api/killswitch", tags=["killswitch"])
state_store = SystemStateStore()


@router.get("/", response_model=KillSwitchResponse)
async def get_killswitch() -> KillSwitchResponse:
    return KillSwitchResponse(active=await state_store.is_kill_switch_active())


@router.post("/activate", response_model=KillSwitchResponse)
async def activate_killswitch() -> KillSwitchResponse:
    await state_store.set_kill_switch(True)
    return KillSwitchResponse(active=True, message="Kill switch activated - all trading halted")


@router.post("/deactivate", response_model=KillSwitchResponse)
async def deactivate_killswitch() -> KillSwitchResponse:
    await state_store.set_kill_switch(False)
    return KillSwitchResponse(active=False, message="Kill switch deactivated - trading resumed")

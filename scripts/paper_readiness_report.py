"""Generate simple paper-to-live readiness gates from recent system state."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.config import get_settings
from trader.db.models.model_version import ModelVersion
from trader.db.models.order import Order
from trader.db.repositories.pnl_snapshots import PnlSnapshotRepository
from trader.logging import setup_logging
from trader.services.system_state import SystemStateStore


@dataclass
class ReadinessReport:
    kill_switch_inactive: bool
    all_champions_present: bool
    enough_pnl_snapshots: bool
    enough_executed_orders: bool
    latest_total_pnl: float
    snapshot_count: int
    executed_order_count: int
    champion_versions: dict[str, str]


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    state_store = SystemStateStore()

    async with factory() as session:
        pnl_repo = PnlSnapshotRepository(session)
        latest = await pnl_repo.get_latest()
        snapshots = await pnl_repo.get_all(limit=1000)
        order_result = await session.execute(select(Order).where(Order.status.in_(("filled", "accepted", "submitted"))))
        executed_order_count = len(order_result.scalars().all())
        champion_result = await session.execute(
            select(ModelVersion).where(ModelVersion.is_champion == True, ModelVersion.is_active == True)  # noqa: E712
        )
        champion_rows = list(champion_result.scalars().all())
        champion_versions = {row.model_type: row.version for row in champion_rows}

        report = ReadinessReport(
            kill_switch_inactive=not await state_store.is_kill_switch_active(),
            all_champions_present=all(model_type in champion_versions for model_type in ("regime", "direction", "magnitude", "filter")),
            enough_pnl_snapshots=len(snapshots) >= 20,
            enough_executed_orders=executed_order_count >= 10,
            latest_total_pnl=float(latest.total_pnl) if latest else 0.0,
            snapshot_count=len(snapshots),
            executed_order_count=executed_order_count,
            champion_versions=champion_versions,
        )

    await engine.dispose()
    print(asdict(report))


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())

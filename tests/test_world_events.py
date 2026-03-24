from world.events import EventScheduler, SimulationEvent


def test_price_change_skips_non_numeric_fields_without_crash():
    scheduler = EventScheduler()
    scheduler.add(
        SimulationEvent(
            day=1,
            type="price_change",
            payload={"service": "delivery", "multiplier": 1.15, "note": "promo"},
        )
    )

    scheduler.process_until(1)
    price_multipliers = scheduler.global_params.get("price_multipliers", {})
    assert price_multipliers.get("delivery") == 1.15


def test_price_change_supports_named_multiplier_pairs():
    scheduler = EventScheduler()
    scheduler.add(
        SimulationEvent(
            day=1,
            type="price_change",
            payload={"delivery": 1.10, "fuel": 1.05},
        )
    )

    scheduler.process_until(1)
    price_multipliers = scheduler.global_params.get("price_multipliers", {})
    assert price_multipliers.get("delivery") == 1.10
    assert price_multipliers.get("fuel") == 1.05

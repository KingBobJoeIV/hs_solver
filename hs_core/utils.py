def parse_effect(effect):
    if effect is None:
        return None, None
    if effect.startswith("deal_"):
        try:
            amount = int(effect.split("_")[1])
            return "deal", amount
        except Exception:
            return effect, None
    return effect, None

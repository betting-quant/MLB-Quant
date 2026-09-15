import math


def american_to_implied_probability(odds: float) -> float:
    odds = float(odds)

    if odds > 0:
        return 100.0 / (odds + 100.0)

    return abs(odds) / (abs(odds) + 100.0)


def probability_to_american(probability: float) -> float:
    probability = float(probability)

    if not 0 < probability < 1:
        raise ValueError("Probability must be between 0 and 1.")

    if probability >= 0.5:
        return -100.0 * probability / (1.0 - probability)

    return 100.0 * (1.0 - probability) / probability


def remove_two_way_vig(over_odds: float, under_odds: float) -> dict:
    over_raw = american_to_implied_probability(over_odds)
    under_raw = american_to_implied_probability(under_odds)

    total = over_raw + under_raw

    return {
        "over_raw_implied": over_raw,
        "under_raw_implied": under_raw,
        "over_no_vig": over_raw / total,
        "under_no_vig": under_raw / total,
        "hold": total - 1.0,
    }


def expected_value(probability: float, american_odds: float) -> float:
    probability = float(probability)
    odds = float(american_odds)

    if odds > 0:
        profit_per_unit = odds / 100.0
    else:
        profit_per_unit = 100.0 / abs(odds)

    return (
        probability * profit_per_unit
        - (1.0 - probability)
    )


def analyze_market(
    model_over_probability: float,
    over_odds: float,
    under_odds: float,
) -> dict:

    market = remove_two_way_vig(over_odds, under_odds)

    model_over = float(model_over_probability)
    model_under = 1.0 - model_over

    return {
        "model_over_probability": model_over,
        "model_under_probability": model_under,

        "market_over_no_vig": market["over_no_vig"],
        "market_under_no_vig": market["under_no_vig"],

        "over_probability_edge":
            model_over - market["over_no_vig"],

        "under_probability_edge":
            model_under - market["under_no_vig"],

        "over_ev":
            expected_value(model_over, over_odds),

        "under_ev":
            expected_value(model_under, under_odds),

        "model_fair_over_odds":
            probability_to_american(model_over),

        "model_fair_under_odds":
            probability_to_american(model_under),

        "sportsbook_hold":
            market["hold"],
    }


if __name__ == "__main__":
    example = analyze_market(
        model_over_probability=0.582,
        over_odds=-115,
        under_odds=-105,
    )

    for key, value in example.items():
        print(f"{key}: {value}")
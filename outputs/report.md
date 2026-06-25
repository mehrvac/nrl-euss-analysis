# Cooling-load temperature sensitivity — Los Angeles

## Scope
- Dataset: NREL ResStock 2024.2 (TMY3), `resstock_tmy3_release_2`
- Geography: Los Angeles (`G0600370`), state CA
- Upgrade: 1 — "ENERGY STAR heat pump with elec backup"
- Buildings sampled: 200 (post-retrofit, with cooling)
- Target: cooling electricity (kWh/day) incl. fans/pumps
- Temperature: outdoor dry-bulb in °F (converted from source °C)
- Modeling resolution: daily | rows (building-days): 73,000

## Headline sensitivity (pooled change-point model)
- **Cooling balance point: 60.0 °F**
- **Sensitivity above balance point: 0.660 kWh per °F per dwelling-day**
- Base load below balance point: 1.041 kWh/day

## Model comparison (held-out test set)

| model       |     r2 |   rmse |    mae |
|:------------|-------:|-------:|-------:|
| linear      | 0.2903 | 4.1513 | 2.5937 |
| changepoint | 0.3087 | 4.0969 | 2.4943 |
| nonlinear   | 0.3187 | 4.0671 | 2.4511 |

## Per-building sensitivity distribution
- Balance point: median 59.8 °F (IQR 57.5–62.5)
- Slope: median 0.577 kWh/°F (IQR 0.325–1.063)

## Figures
- `outputs/figures/load_vs_temp.png`
- `outputs/figures/sensitivity_dist.png`

## Caveat — Catalina Island
ResStock assigns TMY3 weather at the **county** level, so this uses Los Angeles County's representative weather station, **not** Catalina Island's maritime microclimate. Treat the result as a county-level first approximation; substitute a Catalina-specific weather file (re-run with a different weather CSV) for an island-specific estimate.

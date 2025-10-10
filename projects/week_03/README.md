


# GENERATOR

```yaml
# ===================================
# Multi-city example (comment out above, uncomment below)
# ===================================

# cities:
#   - name: Kyiv
#     country: Ukraine
#     bounds: {min_lat: 50.213, max_lat: 50.590, min_lon: 30.239, max_lon: 30.825}
#     center: {lat: 50.4501, lon: 30.5234}
#     zones: [Pechersk, Shevchenko, Podil, Obolon, Darnytsia]
#   
#   - name: Lviv
#     country: Ukraine  
#     bounds: {min_lat: 49.770, max_lat: 49.905, min_lon: 23.930, max_lon: 24.145}
#     center: {lat: 49.8397, lon: 24.0297}
#     zones: [Center, Lychakiv, Sykhiv, Frankivskyi]
#   
#   - name: Berlin
#     country: Germany
#     bounds: {min_lat: 52.338, max_lat: 52.675, min_lon: 13.088, max_lon: 13.761}
#     center: {lat: 52.5200, lon: 13.4050}
#     zones: [Mitte, Charlottenburg, Kreuzberg, Prenzlauer]

# orders_per_minute: 100  # Will be distributed across cities
# vehicles_per_city: 30
```


---

### STREAM

Bloom Filter — швидка перевірка «бачили чи ні» (для дедупа).

HyperLogLog (HLL) — приблизна кількість унікальних (вендори, OD-пари за ґрід-хешем).

Count-Min Sketch (CMS) — оцінка частот для OD-пар.

Misra–Gries — heavy hitters (топ pick-up LocationID у вікні).

Reservoir Sampling — рівномірна вибірка для QA/debug.

Sliding Windows — логіка побудована на «похвилинних» бакетах; можна розширити на 1/5/10 хв.

LSH — простий ґрід-хеш за lat/lon (0.01° ~ 1 км) для агрегації «схожих» OD.
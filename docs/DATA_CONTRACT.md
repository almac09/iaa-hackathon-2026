# Clean data: Week 1 contract

**Where:** `workspace.iaa.<table>_clean`, built by `notebooks/00_load_raw_from_git` then `notebooks/01b_corrections`.
**Different workspace?** Run those two notebooks in your own workspace and you'll get identical tables.
**Status:** reconciled. Every changed cell is logged in `dq_changes`, every removed row in `dq_exclusions`, and the check fails the run otherwise.

## Tables
| Table | Rows |
|---|---:|
| policies_motor_clean / test_motor_clean | 979,968 / 254,986 |
| policies_buildings_clean / test_buildings_clean | 707,726 / 184,155 |
| policies_contents_clean / test_contents_clean | 721,617 / 187,498 |
| claims_clean | 162,546 |
| customers_clean / addresses_clean / weather_clean | 545,240 / 575,983 / 2,879,915 |

## What was corrected
| Rule | Change | Rows |
|---|---|---:|
| R1 | Motor `customer_id` left-padded to 8 characters (train and test). Without this, 10% of motor rows don't join to customers | 123,664 |
| R2 | `exposure` = days on cover ÷ 365 (some rows used days/360 or a raw day count) | 14,164 |
| R3 | `sum_insured` ÷ 1,000 where above 100× the training median (units error) | 3,527 |
| R4 | `construction_type` mapped to brick / stone / timber / non_standard (timber was `T` to 2022, `W` from 2023) | 705,672 |
| R5 | Duplicate claims removed (same claim under an 11-character `claim_id`; mostly 2022, ~3.2m double-counted) | 604 |

Corrected columns now sit at the end of each table.

## Flagged, not changed (see `dq_flags`)
- R6: 413 claims dated before their policy's first row (mostly 2020). They attach to no policy row, so they fall out of any target built by period.
- Still to add (#59): claim `CL30973963` (11.9m motor BI, 10× the next largest); NCB above years licensed (127k motor rows); zero mileage (11.9k); missing `flood_zone` (high-risk addresses, so don't fill with zone 1; see #61).

## Using the data
- **Keep every row**, including lapsed, cancelled and declined. They're real exposure with real claims.
- **Exposure** is in years. Make predictions proportional to it, as an offset or weight (#92).
- **Don't use as features:** `customer_service_contacts` and `renewal_decision` (both record the outcome) and `compulsory_excess` (constant per line). Recommended: keep `written_premium` out too, since it's the incumbent's price, and use it as a benchmark only.
- **Numbers that are really categories:** `flood_zone` (1–5), `vehicle_value_group` (1–20, ordered), `branch_id` (an identifier).
- **The target isn't in these tables.** Attach `claims_clean` to policy rows by `policy_id` with `loss_date` between `period_start` and `period_end`. Costs are reported at 31 Dec 2024 and still need developing to ultimate (#93).
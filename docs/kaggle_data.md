# IAA AI Assisted Hackathon 2026 - Part 1

Create a pricing model for a novel market as well as a pricing tool that can be used by an Actuarial team

![](https://www.kaggle.com/competitions/172537/images/header)

## IAA AI Assisted Hackathon 2026 - Part 1

Submit Prediction

more\_horiz

[Overview](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/overview)[Data](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/data)[Code](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/code)[Models](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/models)[Discussion](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/discussion)[Leaderboard](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/leaderboard)[Rules](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/rules)[Team](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/team)[Submissions](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/submissions)

## Dataset Description

## Files

| File | One row per | Description |
| --- | --- | --- |
| `customers` | Customer | Customer details, shared across all their policies |
| `addresses` | Address | Location and property characteristics |
| `weather` | Address and year | Annual weather and climate measures for each address |
| `policies_motor` | Policy period | Training data for private motor, 2021 to 2024 |
| `policies_buildings` | Policy period | Training data for home buildings, 2021 to 2024 |
| `policies_contents` | Policy period | Training data for home contents, 2021 to 2024 |
| `claims` | Claim | All reported claims on the training policies, as at 31 December 2024 |
| `test_motor` | Policy period | 2025 motor policies to predict |
| `test_buildings` | Policy period | 2025 buildings policies to predict |
| `test_contents` | Policy period | 2025 contents policies to predict |
| `sample_submission` | Test row | A submission in the correct format |

Files are provided in Parquet format.

### How the tables link

-   `customer_id` links policies to `customers`.
-   `address_id` links policies to `addresses` and `weather`.
-   `policy_id` links `claims` to policies. A policy keeps the same `policy_id` across years, so match each claim to a policy row using its `loss_date`.
-   `row_id` identifies each test row in your submission.

A customer can hold motor, buildings and contents policies, and several customers can share an address.

### Policy periods

Each policy row covers one policy for one calendar year, or part of a year. A policy has more than one row in a year if it changed mid-term, for example a new vehicle or a change of address. Rows end early when a policy is cancelled or lapses. `exposure` is the length of the row in years.

## Columns

Identifiers (`customer_id`, `address_id`, `policy_id`, `row_id`, `claim_id`) are text. Read them as text, not numbers.

### customers

| Column | Description |
| --- | --- |
| `customer_id` | Customer identifier |
| `surname` | Customer surname (anonymised) |
| `date_of_birth` | Date of birth |
| `occupation` | Occupation |
| `marital_status` | single, married, divorced, widowed |
| `household_size` | Number of people in the household |
| `credit_band` | Credit score band, A (best) to E |
| `prior_claims_5y` | Claims in the five years before joining, as declared by the customer |
| `licence_points` | Penalty points on the driving licence |
| `years_at_address` | Years the customer has lived at their address |
| `customer_since_year` | Year the customer first took out a policy |
| `email_domain` | Email provider |
| `marketing_consent` | Consented to marketing |
| `document_preference` | email, post or app |
| `previous_insurer` | Insurer before joining |
| `campaign_code` | Marketing campaign at acquisition, if any |
| `branch_id` | Servicing branch |
| `marketing_segment` | Segment assigned by the marketing team |

### addresses

| Column | Description |
| --- | --- |
| `address_id` | Address identifier |
| `region` | Region code, R01 to R12 |
| `urban_density` | rural, suburban, urban, dense\\_urban |
| `flood_zone` | Flood risk zone, 1 (lowest) to 5 (highest) |
| `distance_to_river_km` | Distance to the nearest river |
| `elevation_m` | Elevation above the local river level, in metres |
| `soil_type` | clay, sand, loam, chalk, rock |
| `year_built` | Year the property was built |
| `property_type` | detached, semi\\_detached, terraced, flat |
| `roof_type` | pitched or flat |
| `basement` | Property has a basement |
| `trees_within_5m` | Trees within 5 m of the property |
| `listed_building` | Protected (listed) building |
| `bedrooms` | Number of bedrooms |
| `energy_rating` | Energy efficiency rating, A (best) to G |
| `solar_panels` | Solar panels fitted |

### weather

| Column | Description |
| --- | --- |
| `address_id` | Address identifier |
| `year` | Calendar year |
| `hail_days` | Days with hail in the year |
| `wind_exposure_index` | Exposure to damaging wind, 0 to 10 |
| `freezing_days` | Days below freezing |
| `dry_summer_index` | Summer dryness, 0 (wet) to 1 (very dry) |
| `wildfire_index` | Wildfire risk, 0 to 1 |
| `rainfall_mm` | Annual rainfall |
| `mean_temperature_c` | Annual mean temperature |

### Columns in all policy and test files

| Column | Description |
| --- | --- |
| `row_id` | Row identifier |
| `policy_id` | Policy identifier, constant across years |
| `customer_id` | Policyholder |
| `address_id` | Insured address (for motor, the policyholder's home address) |
| `period_start`, `period_end` | First and last day covered by the row |
| `exposure` | Length of the row in years |
| `voluntary_excess` | Excess chosen by the customer: 0, 250, 500 or 1,000 CU |
| `compulsory_excess` | Excess set by the insurer |
| `channel` | Sales channel: direct, comparison\\_site, broker |
| `payment_method` | annual or monthly |
| `missed_payments_12m` | Missed instalments in the previous 12 months |
| `new_business` | First policy year with the insurer |
| `renewal_count` | Number of previous renewals |
| `days_quote_to_inception` | Days between quote and the start of cover |
| `discount_applied` | Discount given on the premium |
| `written_premium` | Premium charged under the insurer's current pricing |
| `customer_service_contacts` | Contacts with customer service |
| `renewal_decision` | Training files only: renewed, lapsed\\_by\\_customer, cancelled, declined\\_by\\_insurer, mid\\_term\\_change |

### Motor columns

| Column | Description |
| --- | --- |
| `driver_age` | Age of the main driver at the start of the row |
| `years_licensed` | Years the main driver has held a licence |
| `ncb_years` | No-claims bonus years |
| `vehicle_value_group` | Vehicle value group, 1 (lowest) to 20 |
| `vehicle_age` | Vehicle age in years |
| `make_model` | Make and model code |
| `engine_size_cc` | Engine size |
| `body_type` | hatchback, saloon, estate, suv, coupe |
| `transmission` | manual or automatic |
| `seats` | Number of seats |
| `vehicle_colour` | Colour |
| `fuel_type` | petrol, diesel, hybrid, electric |
| `driver_assist` | Fitted with driver-assist features such as automatic braking |
| `cover_type` | comprehensive, third\\_party\\_fire\\_theft, third\\_party\\_only |
| `annual_mileage` | Declared annual mileage |
| `overnight_parking` | garage, driveway, street |
| `vehicle_use` | social, commuting, business |
| `named_young_driver` | A driver under 25 is named on the policy |
| `telematics_opt_in` | Customer opted in to a telematics device |
| `telematics_score` | Driving score from the device, 1 to 100 (blank if not opted in) |

### Buildings columns

| Column | Description |
| --- | --- |
| `policyholder_age` | Age at the start of the row |
| `sum_insured` | Buildings sum insured |
| `construction_type` | Construction: brick, stone, timber, non-standard |
| `occupancy` | owner\\_occupied or let |
| `days_unoccupied` | Longest period the property was unoccupied |
| `smoke_alarms` | Smoke alarms fitted |
| `accidental_damage_cover` | Optional accidental damage cover taken |
| `mortgage` | Property is mortgaged |

### Contents columns

| Column | Description |
| --- | --- |
| `policyholder_age` | Age at the start of the row |
| `sum_insured` | Contents sum insured |
| `tenure` | owner or renter |
| `security_level` | standard, locks, alarm\\_and\\_locks |
| `high_value_items` | Number of declared high-value items |
| `accidental_damage_cover` | Optional accidental damage cover taken |
| `works_from_home` | Policyholder works from home |
| `lodgers` | Lodgers or students in the household |
| `bicycle_cover` | Optional bicycle cover taken |

### claims

| Column | Description |
| --- | --- |
| `claim_id` | Claim identifier |
| `policy_id` | Policy the claim was made on |
| `line` | motor, buildings, contents |
| `peril` | Cause of loss (see below) |
| `loss_date` | Date of the incident |
| `report_date` | Date the claim was reported |
| `status` | closed, open, repudiated (declined) |
| `paid_to_date` | Amount paid so far, net of excess |
| `incurred_to_date` | Paid to date plus the current case estimate for open claims |
| `recoveries` | Amounts recovered from third parties or salvage |
| `repairer_id` | Repairer used (motor damage claims) |
| `third_party_id` | Third party involved (motor third-party claims) |

**Perils.** Motor: `own_damage`, `third_party_prop` (third-party property), `bodily_injury`, `theft`, `glass_hail`. Buildings: `storm`, `flood`, `escape_of_water`, `subsidence`, `fire`, `accidental_damage`. Contents: `theft`, `escape_of_water`, `accidental_damage`, `flood`.

Claims are valued at **31 December 2024**. Some 2024 claims had not been reported by then, and open claims may still develop.

### sample\_submission

| Column | Description |
| --- | --- |
| `row_id` | Row identifier from the test files |
| `prediction` | Expected claim cost for the row, in CU (must be zero or above) |

Submit one file covering every row in all three test files.

## Glossary

| Term | Meaning |
| --- | --- |
| CU  | Currency unit (fictional) |
| Exposure | Time on cover, in years |
| Excess | The part of each claim the policyholder pays |
| NCB | No-claims bonus: years without a claim |
| Sum insured | The maximum the policy pays |
| Incurred | Paid plus outstanding case estimate |
| Ultimate | The final cost of claims once all are reported and settled |
| Repudiated | A claim the insurer declined to pay |
| Comparison site | A price-comparison website |
| Telematics | A device that records driving behaviour |
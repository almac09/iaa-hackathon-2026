<!-- Data audit of the emerging market (A30): six AI analysts, each finding re-derived by a second; synthesis by a third. Aggregates only; identifiers redacted. -->

# Emerging market: what has changed, what carries over, and what to test

**Basis.** This synthesis draws on the checked findings from six audit lenses: quality, portfolio, claims, development, climate and regulation. Where a check corrected a finding, the corrected figure is used. "Fact" means measured in the 2023-24 data or the 2025 test files. "Inference" means an interpretation the data cannot confirm. "Per 1,000" means claims per 1,000 exposure-years, and cost is net of recoveries.

**Summary.** The emerging market is not a scaled copy of the developed one:
- Claims are cheaper (motor severity ×0.61), but motor claims are more frequent.
- Fire is much worse.
- The 2025 book is 77% new business, against 8% in training.

Most of the Part 1 machinery carries over: the target definition, the cleaning rules, the reporting patterns, and the address and climate relativities. Levels, the new-business loading, fire and two hail slopes must be set on emerging data, with credibility towards developed. The new-business loading and buildings fire are the two assumptions most likely to make the 2025 level look wrong in hindsight.

## 1. What has changed, ranked by likely impact on the 2025 prediction

1. **Cost and frequency levels move in opposite directions (fact).**
   - Peril-adjusted severity, emerging against developed:
     - motor 0.612 (95% CI 0.591-0.634);
     - buildings excluding fire 0.675 (0.619-0.737);
     - contents 0.788 (0.724-0.858).
   - Motor ratios are flat from p10 to p90 (0.54-0.74). Sums insured and excesses match.
   - Motor ultimate frequency is 184 per 1,000, against 137.
   - Like-for-like 2023-24 burning-cost ratios: buildings 0.91, contents 0.76, motor 0.79 (0.83 without the 11.89m claim).
   - *Impact:* Part 1 absolute severities would overstate emerging cost by 20-40%, and a single market factor would misprice perils in opposite directions. The cause is probably lower repair and labour costs (inference).

2. **The 2025 book is mostly new business (fact).**
   - 2025 exposure is 3.9 times 2024 (motor 79,007 against 20,104).
   - New business is 76.8-77.0% of 2025 exposure, against 7.7-7.8% in 2023-24. No 2025 new-business customer appears in training.
   - After allowing for mix, new business still claims more. Emerging relativities on frequency and capped cost:

     | Line | Frequency | Capped cost | Claims |
     |---|---|---|---|
     | Motor | 1.103 (SE 0.041) | 1.286 | 697 |
     | Contents | 1.291 | 1.285 | 102 |
     | Buildings | 0.948 | 1.289 | 77 |

   - The developed prior depends on the fitting window. For motor it is 1.121 (frequency) and 1.151 (cost) on 2022-23, but 1.054 and 1.073 on 2023-24.
   - *Impact:* mix plus the new-business loading puts 2025 cost per exposure, before trend, at 1.13-1.30× training for motor, 1.08-1.19× for buildings and 1.04-1.27× for contents. The loading alone accounts for 8-17 points of each range.
   - On every observable, 2025 new business matches 2023-24 organic new business: PSI is 0.033 or less, and the channel split is 69.9/25.0/5.0 against 70.2/25.0/4.9. That a transferred broker block behaves the same way is an inference.

3. **Fire and wildfire (fact).**
   - Fire is 45% of emerging buildings reported cost (185 of 407 per exposure-year), 1.86 times developed.
   - Total losses (incurred at 99% or more of sum insured) run at 0.47 per 1,000 against 0.10, and make up 57.6% of fire cost.
   - Fire cost was 283 per exposure-year in 2023 and 93 in 2024. The fall in claims, from 49 to 21:
     - is not Poisson noise (p = 0.0003);
     - is concentrated in E02 (18 to 2), but also appears outside it (2023 ran at 1.74 times 2024 elsewhere);
     - is unexplained.
   - 22.1% of 2025 buildings exposure has a wildfire index above the developed training maximum of 0.635.
   - *Impact:* fire alone makes the 2023-24 buildings level uncertain by about ±35%. A fit on either year alone gives ×0.50 or ×1.53.

4. **The mix is younger and less experienced (fact, model-based).**
   - 2025 new business against training: age 17-24 is 27.8% of exposure against 19.6%; years licensed of 1 or less 19.2% against 13.0%; NCB 0-2 41.3% against 32.3%.
   - This is new-business mix, not extrapolation: training new business is already 54% aged 17-34. Renewals age, with a mix index of 0.88-0.89.
   - Mix index, test over training: motor 1.086 on capped cost (1.13 against 2024 alone), contents 1.04-1.06, buildings 1.00-1.03.
   - *Impact:* a flat 2023-24 level would understate motor by 8-9% and contents by 4-6%.

5. **The climate differs, and 2025 differs again (fact).**
   - Against developed: hail days ×2.45, wildfire index ×3.4, rainfall +32%, freezing days −49%.
   - 2025 against training is colder, drier and has less hail: freezing 28.9 against 19.3 days, hail 11.4 against 15.6 days, dry-summer index 0.605 against 0.695, rainfall 1,251 against 1,378 mm.
   - 2025 peril multipliers: glass ×0.82-0.87, storm ×0.85-0.92, subsidence ×0.75-0.89, escape of water ×1.10-1.18, fire and flood about ×1.00.
   - *Impact:* small for each line (buildings ×0.99, contents ×1.02, motor ×0.99), but large by region and by row (subsidence ×0.63-1.66 by region).

6. **2024 is immature, and there is no measurable trend (fact).**
   - 2024 motor needs a development factor of 1.19-1.22 (bodily injury 1.63-1.72), depending on the estimator.
   - Reported motor cost per vehicle-year falls 12% from 2023 to 2024 (432.7 to 382.7). After development it rises 2-7%.
   - Mix explains 85-90% of the observed 2023-24 motor frequency fall (−7.2% of −7.9%).
   - Every emerging severity-trend CI includes the Part 1 trends.
   - *Impact:* training on reported 2024 understates motor by about 18% and implies a false negative trend. The Part 1 trend to 2025 adds 5.8-8.1%.

7. **The motor large-claim tail is unobserved (inference from scaling).**
   - Emerging has 10 motor claims above 50k, the largest 155,655.
   - The developed bodily-injury tail scaled by 0.61 matches the emerging exceedances (0.190 against 0.192 at 25k).
   - On that scaling, there is a 14% chance of a claim of about 7m in 2025, worth about 14 per unit of motor exposure (about 3% of loss cost).

8. **The rating information has changed (fact).**
   - Gender is new and predictive. Motor frequency is ×1.32 for men (CI 1.26-1.39). In contents, women cost more (men against women 0.74, CI 0.61-0.90). There is no stable buildings effect.
   - Credit band, exact age, telematics and mileage are gone. Telematics and mileage are the largest loss: D2 +0.0058 excluding the 11.89m claim, against 0.0011 for age banding and 0.0005 for credit.
   - Missingness is about 12% per field against 2.5%, and looks random.
   - Channel mix differs: broker 70% against 25%, comparison site 5% against 35%.
   - Claims handling is unchanged (recoveries, declines, excesses). Expense loads are heavier, but they lie outside the target.
   - *Impact:* mostly on ranking, not level. Gender moves line totals by under 0.4%.

9. **Codes and identifiers differ (fact).** Regions E01-E06 have no R analogue. Construction is coded BRK/MIX/STN/WOD. Occupation has 80 raw values for 20 jobs. IDs collide with developed. *Impact:* silent mis-joins or default levels if the Part 1 code runs unchanged (see section 4).

## 2. Carry over, re-estimate, drop

**Carry over**
- **Target definition and claim filters:** net of recoveries; declined and out-of-period claims excluded; exposure from dates. The recovery and decline mechanics match: own-damage recoveries are 23.6-24.5% of incurred, and declines are 1.30% against 1.05%, all at zero incurred.
- **Part 1 cleaning rules** for exposure, sum insured, year_built and leakage. The redundant steps become assertions.
- **Reporting patterns, especially beyond 365 days.**
  - Two-sample KS tests on 2023 losses give p = 0.15-0.99 on every peril.
  - The emerging pattern would earn only 5.7% credibility for bodily injury and 7.6% for subsidence.
  - In the as-at-2023 backtest, the developed pattern scores 0.988 on motor and 0.948 on bodily injury, against 1.050 and 1.129 for the emerging-only pattern.
  - Keep BF for 2024 bodily injury and subsidence (D-17), and no IBNER uplift (D-28), flagged as untested.
- **Address and climate shapes, as fixed offsets:**
  - flood zone, elevation and river distance: the equal-slope test gives p = 0.34 and 0.36; out of time, developed zones score +127.8/+95.2 against +117.9/−65.4 for emerging-fitted zones;
  - clay × dry-summer for subsidence: p = 0.15; out of time +17.7/+7.0 against +14.7/+2.3;
  - wildfire slope for fire: ×1.19 against ×1.18 per 0.1 (p = 0.89); the emerging slope flips sign between years;
  - freezing-day slope for escape of water (p = 0.80);
  - building-age gradients.
- **Customer-factor shapes, as priors:** prior_claims_5y (motor 1.24/1.74 against 1.26/1.56), years licensed, cover type (third party only 0.48 against 0.44) and age band (17-24 at 1.53 against 1.72).
- **Severity shape**, rescaled, and the bodily-injury GPD shape (0.32-0.50), with the threshold scaled to about 30.5k.
- **Line trends:** +3.8%, +5.3% and +4.2% a year (motor, buildings, contents), so ×1.058, ×1.081 and ×1.064 to 2025. Only the buildings trend is net of weather.
- **Claim-history credibility k**, for renewals only (23.1% of 2025 exposure), rebalanced within renewals.

**Re-estimate on emerging data (with credibility towards developed where stated)**
- **Peril frequency intercepts and severity scale:** well identified (7,048 motor claims) and materially different.
- **New-business factor by line:** it applies to 77% of exposure, the developed prior is window-sensitive, and buildings runs the other way.
- **Fire level and total-loss frequency**, pooled over 2023-24: developed has less than a quarter of the emerging total-loss rate.
- **Glass hail elasticity on log(hail+1):** emerging 0.72-0.73 against developed 0.52-0.53. The difference is significant (p = 0.0004) on 1,345 claims.
- **Storm hail and wind slopes:** the emerging fit beats the developed shape out of time in both directions (+36.4/+35.7 against at best +35.3/+34.3).
- **Escape-of-water level:** ×0.82 of developed at equal climate.
- **NCB, missed payments and payment method:** NCB 0 is 1.73 against 2.56, and missed payments ×1.43 against ×1.32. These partly absorb the missing credit band.
- **Gender**, if adopted. Part 1 has no counterpart.
- **BF priors from emerging 2023**, since a developed prior is 1.2-1.5 times too high for bodily injury, and **large-claim thresholds**.
- **Missing-flood-zone level**, credibility-weighted towards developed.

**Drop**
- **Region factors and any R-to-E mapping.** No E region is a climatic analogue: E region-year hail runs 4-29 days, against an R maximum of 11.5. Region-only factors fail out of time on every peril tested, including non-weather perils (glass −142.0/−38.8; contents theft −1.8/−4.0).
- **customer_service_contacts and renewal_decision:** outcome leakage.
- **Credit band, exact-age splines, telematics and mileage:** absent from the emerging data.
- **Duplicate encodings of new business:** raw customer_since_year (2025 on every new-business row, outside the training range), campaign_code presence, discount_applied, and renewal_count beyond a 0/1+ split.
- **Part 1 missing-level relativities for the fields at about 12%.** A single soil-missing factor of ×1.66 would overprice E01 by 37% and underprice E06 by 13%.
- **Part 1 currency-based levels:** absolute severities, the 50k large-claim threshold and the 1m BF cap.
- **Emerging-only development patterns and an emerging-only trend.**

## 3. Modelling adjustments to test, ranked

**Validation rules for every item:**
- Develop 2024 before scoring it. Otherwise every test rewards under-prediction.
- Score in both directions: fit 2023 and score 2024, then the reverse.
- State the D2 baseline, because two definitions are in use (unweighted mean, and constant rate × exposure). Add customer-cluster bootstrap intervals.
- The holdout year is only 9.1% new business and has different weather. It therefore tests ranking and peril shapes, not the 2025 level. Level choices need a developed out-of-time check (fit 2021-23, score 2024) and sensitivities.

1. **A credibility-blended new-business factor per line.**
   - *What:* blend the developed prior with emerging by Z = n/(n+k). k = 1,000-1,500 is an assumption, giving motor Z of 0.3-0.4 and home lines 0.05-0.1. Include an inception-month control: 2025 motor new business has a median inception day of 9, against 188-193 in training.
   - *Validate:* A/E for new business and renewals separately on 2024. Emerging 2024 new-business A/E is 1.019 (CI 0.92-1.13), against 1.209 in 2023. Repeat on developed to test the prior's stability (motor fell from 1.093 to 1.022).
   - *Risk:* 2024 new business is 3-5% under-reported, broker selection cannot be observed, and the result moves 77% of exposure.

2. **A peril rebase with developed relativities as offsets.**
   - *What:* refit only the frequency intercepts and a severity scale by peril (motor 0.61, buildings non-fire 0.675, contents 0.79).
   - *Validate:* peril A/E on 2024 from a 2023 fit, D2 against a free emerging GLM, and reconciliation to the 2023-24 control totals of 415.8, 88.9 and 445.1 per exposure-year (buildings, contents, motor).
   - *Risk:* the motor base ranges from 1.06 to 1.20 with the weather specification, and the intercept absorbs that error.

3. **A separate buildings fire model.**
   - *What:* total-loss frequency × sum insured (capped at sum insured) driven by wildfire, plus partial losses at the developed shape × 1.1-1.24. Extrapolate the developed wildfire slope beyond 0.635, with a clamped variant. Pool the level over 2023-24.
   - *Validate:* 2024 fire frequency is 40% of 2023's (1.74 against 4.31 per 1,000), so a 2023 fit will fail on level. Report that as evidence of volatility, test the slope out of time on developed, and check E02 2023 against the weather table before pooling.
   - *Risk:* the model rests on 70 claims. Extrapolation adds 6.7% to fire frequency overall and 24.7% in E02.

4. **Parametric weather terms, scored on 2025 weather.**
   - *What:*
     - glass on log(hail+1), with the elasticity weighted towards emerging;
     - storm on emerging or blended slopes;
     - subsidence on the developed clay × dry-summer offset;
     - escape of water on the developed freezing slope, with an emerging level;
     - flood on frozen developed zones, with rainfall capped at the developed p99 (1,787 mm) or applied only where flood zone is missing.

     Clip at the training minimum and maximum, not at p1-p99. Only 0.04-0.40% of 2025 values fall outside the minimum and maximum, while 10-14% fall outside p1-p99.
   - *Validate:* cross-year deviance in both directions (glass: developed-fixed +49.8/+119.4 against emerging +39.6/+124.9), and predicted against observed 2023-24 change by peril (storm +14-20% against +13.7%; glass +12.6% against +16.2%).
   - *Risk:* 2025 sits in sparse tails. Every developed curve under-predicts the band with 30 or more hail days (A/E up to 1.39), and log-linear extrapolation in hail days over-predicts glass (A/E 0.70-0.81).

5. **Development method.**
   - *What:* pick one estimator. The days-averaged estimator and Part 1's claim-weighted sql/10 differ by 0.25m, 2.7% of 2024 motor. Use developed patterns on fast perils, and BF on 2024 bodily injury and subsidence with an emerging 2023 prior. Cross-check with frequency × severity. Optionally net open own-damage and theft claims of expected recoveries (−1.2% on 2024 motor).
   - *Validate:* the as-at-2023 backtest (motor 0.988; bodily injury 0.948 on cost and 1.039 on counts).
   - *Risk:* the tail beyond 730 days (2.55% of bodily-injury cost) is untested. Open bodily-injury claims average 1.19-1.24 times closed ones, against 0.98-1.04 in developed. If emerging case reserves are prudent, the 2024 bodily-injury ultimate is overstated, so this risk runs one way only.

6. **Mix and trend through the model.**
   - *What:* no flat level. Apply the trend factors (×1.058, ×1.081, ×1.064) and use the mix indices as reasonableness checks: motor 1.08-1.09, contents 1.04-1.06, buildings 1.00-1.03.
   - *Validate:* the model reproduces the 2023-24 motor frequency change from mix alone (−7.2% against −7.9% observed).
   - *Risk:* the Part 1 motor and contents trends are not net of weather.

7. **A motor large-claim load.**
   - *What:* the Part 1 GPD with a threshold of about 30.5k, shape 0.32-0.50 as a sensitivity, and large-claim frequency ×1.4. Buildings large losses go through the fire model. Contents gets no split (one claim above 50k).
   - *Validate:* only the exceedance match across thresholds can be tested. 25 excesses over 30.5k in two years cannot support an out-of-time test.
   - *Risk:* the tail parameters cannot be tested, and an expected-value load hides an all-or-nothing outcome.

8. **Gender: in motor, shrunk in contents, not in buildings, with a gender × age-band term for frequency only.**
   - *Validate:* the motor out-of-time D2 gain is +0.0049 (CI +0.0009 to +0.0085) from 2023 to 2024, and +0.0026 (CI −0.0052 to +0.0095) in reverse. The contents intervals include zero. The gender × age term helps frequency (+0.0017 to +0.0023) but not cost (−0.0007 to −0.0041).
   - *Risk:* governance. Gender pricing has been unlawful in the UK and EU since the Test-Achats ruling (21 December 2012), and the emerging market's rule is not stated. If gender is excluded, occupation recovers about 21% of the motor effect as a proxy. Run both versions.

9. **Missing-value treatment.**
   - *What:* impute, or set to base, for the fields at about 12%. For soil, weight by the regional probability of clay. Keep missing flood zone as its own level.
   - *Validate:* out-of-time D2 against treating missing as a level, a severity check by missingness, and the days_unoccupied exception (35.4 against 46.3 per 1,000, 2.3 standard errors).
   - *Risk:* missingness may carry information on severity, which is untested.

10. **Log written premium as an offset or prior.**
    - *What:* the emerging tariff appears to be the developed tariff carried across, since 2024 premium by age band matches within about 1% (inference).
    - *Validate:* it already beats the full ridge GLM in 5 of 6 out-of-time comparisons.
    - *Risk:* this conflicts with Part 1 decision D-05 and with the quality lens's advice to keep premium out of the models. The premium is gender-neutral and sits at the developed level, so it would need re-levelling (about 0.73-0.78 of tariff after development, an inference). It also overlaps the rating factors.

11. **Region: none, against a heavily shrunk factor on top of the address and weather terms.**
    - *Validate:* cross-year deviance.
    - *Risk:* losing a genuine regional effect such as E02 fire, though the weather terms should carry most of it.

## 4. Data fixes before modelling

1. **Namespace every key by market** before any union, assert uniqueness, and never join emerging tables to developed tables or caches by ID. 126,625 of 209,063 test rows (60.6%) share a row_id with developed training, and 131,946 (63.1%) collide on a row, customer or address ID.
2. **Recompute training exposure** as (end − start + 1)/365 and keep the raw value.
   - About 605 rows are wrong (604-609, depending on tolerance): 114 are recorded in days and the rest are on a 360-day basis.
   - Raw totals are 34-44% too high, almost entirely because of the day-count rows.
   - Assert that the test files need no change: 0 of 209,063 rows deviate.
3. **Divide sum insured by 1,000 where it is 1m or more**, on both lines and in both training and test. That covers 53 buildings and 57 contents training rows, and 104 and 135 test rows (239). The gaps are clean: nothing between 894k and 60m, or between 283k and 7.6m.
4. **Replace year_built 1900** with the typical year for the energy rating, and flag it. This affects 4,421 addresses (3.23%), including 1,673 buildings and 2,151 contents test rows. Give the 521 without a rating a neutral age.
5. **Map construction codes:** BRK to brick, STN to stone, WOD to timber, MIX to non_standard. The mapping is inferred from matching shares (69.9/10.2/7.8/12.1 against 70.0/10.0/7.9/12.1) and the same frequency order. Assert that no level is left unmapped.
6. **Normalise occupation** with lower(trim()) and strip the "self-employed " prefix, which takes 80 values to 40 and then 20. Without this, half the non-missing values fail to match the Part 1 levels.
7. **Exclude customer_service_contacts and renewal_decision.** Add a guard that rejects any feature whose test mean matches the non-claimant subgroup rather than the training mean.
8. **Keep the claim filters.**
   - The period-match filter removes 36 claims (97,100 incurred), 34 of them dated before the insurer entered the market.
   - The repudiation filter removes contents 13 and motor 107.
   - Deduplication becomes an assertion of zero removed.
   - The resulting targets are 1,056 buildings, 998 contents and 7,048 motor claims.
9. **Collapse the tenure fields** into one new-business indicator.
10. **Treat the missing values** as in item 9 of section 3, and do not route them to the Part 1 missing levels.
11. **Develop the 2024 target** and fix the estimator before any fit.
12. **Turn the no-op Part 1 steps into assertions:**
    - customer_id padding (all IDs have 8 characters);
    - R11 elevation (there is no R11, and every E region caps at 60.0 m);
    - deduplication;
    - licensed before 16. The banded check is weak and does not prove the problem is absent.
13. **Leave as recorded:**
    - NCB above years licensed (16.4% of training, 22.1% of test);
    - claims marginally above sum insured (at most 1.0004×, plus one contents claim at 1.074×);
    - regional written-premium ratios, which are probably rate relativities.
14. **Pricing tool only:**
    - The comparison-site commission is missing (10,458 test rows, 5.0%), so state an assumption.
    - Apply the fixed expense once per policy, not per row (motor has 90,521 rows on 83,907 policies).
    - The 20 reference policies are exact copies of test rows. Use them for the premium build-up, with and without gender.

## 5. Main uncertainties to communicate

Present a central estimate per line, with the items below as named sensitivities rather than one combined interval. They are not additive.

1. **New-business uplift, the largest.** Before trend, the range is 1.13-1.30× training for motor, 1.08-1.19× for buildings and 1.04-1.27× for contents. Show the developed-only and emerging-only loadings as the low and high cases. Add a broker-selection sensitivity of ±5-10% on new business, which is an assumption, not a data finding.
2. **Buildings fire.**
   - Fire alone makes the buildings level uncertain by about ±35%.
   - About 22 total losses are expected in 2025.
   - The cause of the 2023 spike is unknown.
   - Extrapolating rather than clamping the wildfire term moves fire frequency by 6.7% overall and by 21-25% in E02 and E04.
3. **Motor base frequency specification:** 1.06-1.20 depending on the weather terms. The quoted sampling CIs (for example 1.16-1.21) understate this.
4. **Motor tail:** a 14% chance of a bodily-injury claim of about 7m, worth about 3% of loss cost in expectation. The GPD shape is uncertain between 0.32 and 0.50.
5. **Trend:** the emerging data cannot measure it. The motor severity step is −4.9% (CI −13.1% to +5.7%). Show sensitivities at 0% and at twice the Part 1 trend.
6. **Development:**
   - 2024 motor carries ±2.9% (one standard deviation), or ±1.5% on the pooled 2023-24 level.
   - 2024 home lines carry ±1.5-2.5% (±1% pooled).
   - The estimator choice is worth 2.7% of 2024 motor.
   - Subsidence methods range from 0.31m to 0.51m, up to 4.6% of 2024 buildings.
   - The tail beyond 730 days is untested, and open bodily-injury reserves may be prudent.
7. **2025 weather in sparse tails.** 10-14% of 2025 address-years fall outside the training p1-p99 for freezing days, rainfall and temperature, against about 2% by construction. Peril effects range from ×0.75 (subsidence) to ×1.18 (contents escape of water), and regional subsidence from ×0.63 to ×1.66, while line totals move by 2% or less.
8. **Sampling error alone** (90% bootstrap, 2023-24 cost per exposure-year): buildings 351-484, contents 80-98, motor 420-471.
9. **Decisions rather than uncertainty in the mean:**
   - Gender moves line totals by under 0.4%, but individual motor rows move by −11% to −13% for men and +13% to +17% for women.
   - Marketing segment is a credit proxy: in the developed data it maps exactly onto credit tiers.
   - Occupation is a partial gender proxy.
10. **Inferences the work relies on, to be stated as such:**
    - the construction mapping;
    - that the broker book behaves like organic new business;
    - that the tenure fields describe a seasoned or acquired book (88% of 2023 rows are renewals in the insurer's first year in the market);
    - that the tariff was carried across from the developed market.

**Open decisions for the team:**
- which development estimator to use;
- which window to use for the developed new-business prior;
- whether written premium enters the model;
- whether gender is used, and on what legal basis.
# IAA AI Assisted Hackathon 2026 - Part 1

Create a pricing model for a novel market as well as a pricing tool that can be used by an Actuarial team

![](https://www.kaggle.com/competitions/172537/images/header)

## IAA AI Assisted Hackathon 2026 - Part 1

Submit Prediction

more\_horiz

[Overview](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/overview)[Data](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/data)[Code](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/code)[Models](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/models)[Discussion](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/discussion)[Leaderboard](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/leaderboard)[Rules](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/rules)[Team](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/team)[Submissions](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1/submissions)

## Overview

# **Challenge: Actuarial Pricing Across Markets with AI Assistance**

You are part of the pricing team of an established insurer operating in developed insurance markets.

Your company is reviewing its technical pricing approach across three major lines of business. Your task is to develop claim prediction models that are technically sound, adaptable, explainable, and practical to use.

Your solution must demonstrate an AI-assisted actuarial workflow.

The underlying claim prediction model itself does not need to use AI – you may use traditional actuarial techniques, machine-learning models, or a combination of approaches. We are interested in how AI can work alongside actuaries throughout the problem-solving process.

Start

3 days ago

Close

4 days to go

### Description

link

keyboard\_arrow\_up

# Week 1: Build Your Pricing Approach

At the start of the challenge, you will receive a synthetic insurance data from an established market. The data covers three lines of business: private motor, home buildings and home contents. It runs over four policy years (2021 to 2024) in a developed market.

Your task is to predict the expected ultimate claim cost for each policy in 2025 : the final cost of all claims with a loss date in the row’s period, net of the policy excess and net of recoveries, excluding declined claims.

You are free to decide:

-   how to explore and prepare the data;
-   which modelling techniques to use;
-   how to structure your pricing workflow;
-   where AI should support your work;
-   how your models should be tested and validated.

Your technical model could use familiar actuarial approaches such as GLMs, newer machine-learning techniques, or combinations of both.

# Week 2: Expansion

In the second week, your insurer is considering expansion.

A new dataset will be released. Link will be provided at the end of Week 1.

The data may contain different characteristics and practical challenges.

Your task is to determine how your existing modelling approach should adapt.

We want to see how your team identifies what has changed, responds to those differences, and communicates any remaining uncertainty.

### Evaluation

link

keyboard\_arrow\_up

Submissions will be evaluated against the following requirements:

| Category | Weight | Key Requirements |
| --- | --- | --- |
| 1\\. Pricing Performance & Adaptation | 40% | • Quantitative accuracy across both Week 1 & 2 dataset. • Appropriate modelling choices and adaptation to emerging market differences. • Clear treatment of data quality, limitations, and remaining uncertainties. |
| 2\\. Effective Use of AI | 25% | • Meaningful integration of AI throughout the actuarial workflow (e.g., EDA, feature engineering, modeling, diagnostics, and testing). • Demonstrated improvement in quality, efficiency, challenge, or reliability. • Rigorous verification of AI-generated outputs. |
| 3\\. Trust, Validation & Guardrails | 20% | • Comprehensive validation checks, automated tests, and model diagnostics. • Governance controls, reproducibility, and human review/escalation protocols. |
| 4\\. Communication & Documentation | 15% | • Clear and concise video demo and written report explaining methodology and key decisions. • Well-structured code repository with version control. |

### Submissions

link

keyboard\_arrow\_up

## What should you submit after 2 weeks (Part 1 + Part 2)

Each team should provide at the end of 2 weeks (in a final submission form - link to be provided)

-   Team name, members name
-   Week 1's prediction file (in addition to submitting to Kaggle leaderboard in Part 1)
-   Week 2's prediction file (in addition to submitting to Kaggle leaderboard in Part 2)
-   A short video demo explaining your approach, results, AI-assisted workflow, and how your solution changed when the new market data was introduced. Only first 5 minutes will be graded.
-   A concise write-up describing the modelling approach, major decisions, use of AI, assumptions, limitations, adaptation to the new market, and guardrails. Only first 5 pages will be graded.
-   Your code, using scripts and/or notebooks, maintained using version control so that the development of the solution can be understood and reproduced.

## How Submissions Will Be Reviewed

Submissions will be assessed using a combination of quantitative model evaluation and qualitative review. AI-assisted review tools will be used as a first layer to screen videos, documentation, and code.

Final assessment and judging decisions remain the responsibility of the judging panel.

## Submission file

For each row\_id in test\_motor, test\_buildings and test\_contents, predict the expected ultimate claim cost for that row in fictitious currency unit (the final cost of all claims with a loss date in the row’s period, net of the policy excess and net of recoveries, excluding declined claims). This is the risk premium, not the premium charged.

Submit one CSV covering all three files, with a header:

> row\_id, prediction
> 
> M02409311,583.47
> 
> B01834522,412.10

Predictions must be zero or above. See sample\_submission.csv.

## Accuracy of submissions

Predictive accuracy is evaluated using the Tweedie D² score (power = 1.5), which measures how well the prediction match actual claim, relative to a constant-rate baseline. Actual claim cost is the ultimate cost of claims with a loss date in the row’s period, net of excess and recoveries, excluding declined claims. Higher scores indicate better predictive performance.

### Prizes

link

keyboard\_arrow\_up

Top 3 teams will receive prize sponsored by IAA.

-   1st prize - CAD $500
-   2nd prize - CAD $100
-   3rd prize - CAD $50

They will also be invited to share their experience and learnings via subsequent IAA webinars.

Please note that the prize will only be awarded to members or students of any Full or Associate Member Association (MA) of the IAA. The IAA reserves the right to request an MA to confirm the applicant’s membership.

Prizes are decided after Part 2 by the judging panel, based on the evaluation criteria above, not on the Part 1 leaderboard alone.

### Citation

link

keyboard\_arrow\_up

nightowl and ShaunBennet. IAA AI Assisted Hackathon 2026 - Part 1. https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1, 2026. Kaggle.
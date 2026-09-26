# Data profile: Amazon Books reviews

- **Rows:** 10,000 (one row per review)
- **Columns:** `rating`, `title`, `text`, `asin`, `parent_asin`, `user_id`, `timestamp`, `helpful_vote`, `verified_purchase`, `review_date`
- **Review dates:** 1998-05-05 to 2023-03-16
- **Distinct books:** 9,551 (top 10 books hold 0.3% of reviews)

## Labels

| Sentiment | Rule | Reviews | Share |
|---|---|---|---|
| positive | 4-5 stars | 8,424 | 84.2% |
| neutral | 3 stars | 1,066 | 10.7% |
| negative | 1-2 stars | 510 | 5.1% |

| Stars | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Reviews | 150 | 360 | 1,066 | 2,597 | 5,827 |

## Text length

| Words per review | min | p5 | median | mean | p95 | max |
|---|---|---|---|---|---|---|
| | 1 | 6 | 134 | 193.9 | 546 | 4091 |

## Completeness and duplicates

| Column | Null rate |
|---|---|
| `rating` | 0.0% |
| `title` | 0.0% |
| `text` | 0.0% |
| `asin` | 0.0% |
| `parent_asin` | 0.0% |
| `user_id` | 0.0% |
| `timestamp` | 0.0% |
| `helpful_vote` | 0.0% |
| `verified_purchase` | 0.0% |
| `review_date` | 0.0% |

- Empty review text: 0
- Rows with text identical to another review: 405
- Same reviewer reviewing the same book twice: 5

## Helpful votes

- 53.3% of reviews have zero helpful votes
- Median 0, p95 12, max 883
- Verified purchases: 31.4%

## Flags

- 405 rows share identical text with another review (short stock phrases or copy-pasted reviews)
- Class imbalance: largest class is 16.5x the smallest; use stratified splits, class weights and macro-averaged metrics
- 53.3% of reviews have no helpful votes, so votes are a weak credibility signal on their own
- 216 reviews have fewer than 3 words

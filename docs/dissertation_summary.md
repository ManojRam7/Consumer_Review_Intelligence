# Dissertation summary

**Machine learning-based sentiment analysis for marketing optimisation**
MSc Data Science, Coventry University (7150CEM). Supervisor: Prof. Roohi Barzamini.

## Research question and objectives

*How can sentiment analysis of social media data improve the effectiveness of marketing campaigns?*

1. Classify public sentiment in Amazon book reviews as positive, negative or neutral.
2. Compare basic machine learning methods (Naive Bayes, Random Forest, Support Vector Machines).
3. Identify trends and patterns in consumer opinion that can shape marketing strategy.
4. Assess how sentiment, including misleading or contradictory reviews, affects campaign decisions.
5. Recommend how marketers can use the findings.

Amazon reviews were used as a structured proxy for social media feedback, on the supervisor's
advice, because they pair free text with a star rating that can validate the sentiment label.

## Data

- **Source:** Amazon Reviews 2023 (McAuley Lab), `raw_review_Books`, loaded from Hugging Face.
- **Sample:** the first 10,000 reviews (`dataset["full"][:10000]`).
- **Fields used:** review text, rating (1-5), timestamp, helpful votes, anonymised user ID.
- **Labels:** positive for 4-5 stars, neutral for 3, negative for 1-2.

## Method

| Step | Detail |
|---|---|
| Cleaning | Missing text or ratings removed, lowercasing, special characters, punctuation and numbers removed, stop words removed |
| Normalisation | Slang expanded (`gr8` to `great`), emojis converted to text, stemming and lemmatisation, tokenisation |
| Features | Bag-of-Words, TF-IDF, VADER polarity scores, text length, n-grams |
| Split | 80% training, 20% testing; resampling to address class imbalance |
| Models | SVM, Multinomial Naive Bayes, Random Forest. Logistic Regression was tried as a baseline and dropped after weaker preliminary results |
| Tuning | `GridSearchCV` over `C`, kernel, gamma and degree for the SVM: 3 folds x 48 candidates = 144 fits |
| Evaluation | Accuracy, precision, recall, F1, confusion matrices |
| Credibility | Reviews labelled credible or inconsistent from rating-text contradiction, helpful votes and generic content; sentiment compared across the two groups with `pd.crosstab` and pie charts |
| Exploration | Word clouds for positive, neutral and negative reviews, top Bag-of-Words terms, top TF-IDF terms |

## Reported results

| Model + features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM + TF-IDF (C = 0.1, linear kernel, gamma = scale) | 0.85 | 0.85 | 1.00 | 0.92 |
| Naive Bayes + TF-IDF | 0.85 | 0.85 | 1.00 | 0.92 |
| Naive Bayes + BoW | 0.60 | 0.91 | 0.59 | 0.71 |
| Random Forest + BoW | 0.50 | 0.50 | 1.00 | 0.67 |
| Random Forest + TF-IDF | 0.50 | 0.50 | 1.00 | 0.67 |
| SVM + BoW (tuned) | 0.03 | | | |
| Naive Bayes, no feature weighting | 0.03 | | | |

Precision, recall and F1 are for the majority class. The per-class reports showed the minority
class was rarely predicted correctly, and the evaluation samples were small (20 test reviews in the
TF-IDF runs).

## Conclusions

- **Feature representation had a larger effect than the choice of model.** TF-IDF outperformed
  Bag-of-Words for every classifier; the SVM went from 0.03 with BoW to 0.85 with TF-IDF.
- **SVM + TF-IDF performed best**, consistent with margin-based models suiting sparse,
  high-dimensional text.
- **Random Forest was the weakest** (0.40 to 0.50), with little gain from better features.
- **Class imbalance hurt minority classes**; SMOTE or class weighting were recommended.
- **Contradictory reviews distort sentiment** and can erode trust in a brand if left unmonitored.
- **Transformers** (BERT-style models) were identified as the next step for sarcasm and context.

## Business implications

1. Sentiment tracking gives near-real-time feedback on campaigns and product changes.
2. Positive themes can be reinforced in messaging; negative themes point to product or service fixes.
3. Sentiment segments support targeting: loyalty offers for promoters, recovery for detractors.
4. Detecting misleading or contradictory reviews protects brand reputation.

## How this repository extends the dissertation

| Dissertation limitation | What the code does |
|---|---|
| Small evaluation samples | All six model and feature pairs are trained and tested on the full 10,000 reviews (80/20 stratified) |
| Minority classes missed | Balanced class weights (or `--oversample`), model selection on macro F1, per-class reports |
| Accuracy read without context | A majority-class baseline is reported next to every model |
| Vectoriser fitted before the split | Vectorisers sit inside each pipeline, so every cross-validation fold learns its own vocabulary |
| Subjective credibility criteria | Written, testable rules with thresholds in `config.py` and unit tests |
| Negation lost in stop-word removal | Negations such as *not* and *never* are kept |

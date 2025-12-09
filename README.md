# ABSA Yelp Review Analysis  
By Rachel Yu

### Project Overview
This project builds an Aspect-Based Sentiment Analysis (ABSA) system to transform unstructured Yelp reviews into structured insights. Instead of relying on a single star rating, the system extracts sentiment for specific aspects—**service, food, ambience, and pricing**—to reveal what truly drives customer satisfaction.

### Introduction / Motivation
Star ratings often hide important nuances within reviews. A customer may love the food but dislike the service, yet both sentiments collapse into the same overall rating. To address this limitation, this project applies ABSA to Yelp reviews, allowing sentiment to be analyzed at the **aspect level**. This helps businesses understand which components of their experience most influence customer perception and where improvements will have the greatest impact.

### Methods
The project pipeline consists of several key stages: data preprocessing, weak labeling, model development, and aspect aggregation.

#### Data Preprocessing
- Utilized the Yelp Open Dataset including review text, star ratings, and business metadata.  
- Split reviews into individual sentences to isolate single opinions.  
- Implemented keyword-based lexicons to detect the four target aspects: service, food, ambience, and pricing.

#### Weak Labeling
- Derived sentence-level sentiment labels from the overall star rating:  
  - High stars → positive sentiment  
  - Low stars → negative sentiment  
- Enabled supervised model training without manual annotation.

#### Model Development
Three sentiment classification models were trained and evaluated:
- **Logistic Regression (TF-IDF)** baseline  
- **CatBoost** with tuned hyperparameters  
- **DistilBERT** fine-tuned for sentence-level sentiment

**Best Model: DistilBERT**
- Macro F1: **0.869**  
- Accuracy: **0.89**

#### Aspect Aggregation and Rating Correlation
- Computed positive and negative sentiment shares for each aspect.  
- Derived an aspect sentiment index using:  `positive_share − negative_share`  
- Aggregated aspect sentiment to the business level.  
- Ran correlation and OLS regression against business star ratings to identify which aspects most influence satisfaction.

### Result
The ABSA pipeline successfully converted thousands of Yelp reviews into structured aspect-level insights.  
- **DistilBERT outperformed other models**, achieving strong sentiment classification metrics.  
- Aspect-level analysis revealed that:  
  - **Service** and **food** sentiment correlate most strongly with overall star ratings.  
  - Ambience and pricing had comparatively weaker effects.  
This demonstrates the value of ABSA for revealing hidden patterns in customer feedback beyond aggregate ratings.

### Conclusion
This project shows that Aspect-Based Sentiment Analysis is an effective tool for uncovering nuanced insights from large-scale review data. By breaking sentiment down by aspect, businesses can better understand what drives customer ratings and target improvements more strategically. Future extensions may include expanding aspect categories, refining lexicons, or leveraging larger transformer models for improved accuracy.


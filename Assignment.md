General Info
Deadline: June 14th, 11:55 PM 

Late Penalty: 5% deduction per day

Submissions made after June 20th, 23:55 PM will receive 0 points

Submission Format: (Submit by Group)

[Through Kaggle] A CSV file with top-10 recommendations per user (see detailed submission format in the Kaggle link).

[Through Brightspace] A zip file with executable code and README file.

[Through Brightspace] A technical report (PDF) describing your methodology, trade-offs, and error analysis (Maximum 6 pages without reference).

Kaggle Link: https://www.kaggle.com/t/74b6349c792d42c6b41c1044083927e0
❗❗❗There is no retake option for the Assignments. Please manage your time wisely.

Task
In this final project, you are required to design and implement a complete recommender system using a unified user-item interaction dataset. Your model will be evaluated on ranking-based metrics using a hidden test set. All students/teams will compete on a public leaderboard hosted on Kaggle.

You are encouraged to explore and combine a variety of modern recommendation techniques. Your focus should be on both model performance and engineering rigor — from preprocessing to inference and analysis.


Learning Objectives
Apply and integrate key recommendation algorithms (collaborative, sequential, graph-based, neural, hybrid)

Practice end-to-end model development: preprocessing, modeling, training, inference, and evaluation

Analyze trade-offs, identify model weaknesses, and communicate results clearly

Competition Rules
1. Eligibility
This competition is open only to students enrolled in the course.
Each submission must be made by a registered individual or team.
2. Team Formulation
Each team must use only one Kaggle account for submissions.
Your team name and members must match exactly with your team on Brightspace.
You are not allowed to collaborate across teams or share models or predictions.
3. Data Usage
You may only use the data provided in the Data tab (train.csv, test.csv, item_meta.csv).
External data is not permitted.
You are free to perform any preprocessing, feature engineering, or augmentation on the provided data.
⚠️ You are strictly prohibited from training your model on any data other than the provided training set. This includes, but is not limited to, pretrained embeddings derived from external datasets, public test sets, or any leaked ground-truth labels. Any team found violating this rule will receive a score of ZERO for the entire group, with no exceptions.
4. Academic Integrity
All code and reports must be written by you or your teammate.
You may use open-source libraries as long as they are properly cited.
Plagiarism, code sharing between teams, or copying models from external solutions is strictly prohibited and will result in disqualification and academic consequences.
Project Requirements
Modeling Flexibility:

Use any combination of techniques or propose your own models. Examples include:

Matrix factorization (e.g., NMF, SVD)

Neural Collaborative Filtering (NCF)

Graph-based methods (e.g., LightGCN)

Transformer-based models (e.g., BERT4Rec)

Hybrid and content-aware recommenders

Reproducibility:

Your entire pipeline must be executable and reproducible by instructors.

Evaluation Breakdown
Component

Weight

Description

Leaderboard Ranking

30%

Based on public leaderboard score using Recall@10

Technical Quality

30%

Assesses code structure, modeling rigor, and efficiency

Technical Report

40%

Evaluates depth of understanding and analysis quality

1. Leaderboard Ranking (30%)
Your team’s performance on the public leaderboard will contribute to your final score.

Leaderboard Rank

Score Contribution

Top 10%

100%

11–30%

90%

31–50%

80%

51–70%

70%

71–100%

60%

2. Technical Quality (30%)
Sub-Criterion

Description

Code Completeness

Code runs from end-to-end with clear README/instructions

Efficiency

Resource usage is reasonable; training/inference time is optimized where possible

Code Readability & Style

Code is well-structured, modular, and uses consistent naming

Modeling Technique

Chosen methods are justified, and demonstrate a clear understanding of recommender system concepts

Engineering Rigor

You perform sensible evaluation, model selection, and tuning

3. Technical Report (40%)
Your report should be clear, concise, and demonstrate analytical thinking. It should be structured with the following components:

Section

Description

1. Dataset Preprocessing

Explain how you handled missing values, timestamps, duplicates, etc.

2. Model Design

Describe your architecture(s), embedding strategy, loss functions, hybrid structure, etc.

3. Training Procedure

Describe training process: data splits, sampling strategies, negative sampling, etc.

4. Inference Pipeline

Explain how top-k items are ranked for each user efficiently

5. Hyperparameter Analysis

Show your tuning process and rationale for final settings

6. Performance Analysis

Analyze evaluation results, failure cases, and possible reasons for poor performance

7. Reflections

What worked well? What would you try next?

Note:

1. Please use ACM double-column format (https://www.overleaf.com/latex/templates/association-for-computing-machinery-acm-sig-proceedings-template/bmvfhcdnxfty).

2. Write a concise 3–4 page report (do not exceed 6 pages excluding references)
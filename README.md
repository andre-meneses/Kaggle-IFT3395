# Compétition Kaggle - IFT 3395

## Introduction

## To-do 
**1. Implement Logistic Regression:**
   - [x] Data Preprocessing:
     - [x] Split data into features and labels.
   - [ ] Model Building:
     - [ ] Implement the logistic regression model.
   - [ ] Training:
     - [ ] Implement the loss function.
     - [ ] Set up the training loop.
       - [ ] Perform weight updating during training.
   - [ ] Inference:
     - [ ] Implement the inference function for predictions.

**2. Explore Alternative Classifiers:**
   - [ ] Research and implement an alternative classification model.

## Ideias 
- Apply Principal Component Analysis (PCA) to the features
- k-fold training. 

## Git Commit Guideline

### Commit Structure
Each commit message should consist of a brief description of the changes made, followed by an optional extended description. Use imperative mood (e.g., "Add feature" instead of "Added feature").

### Commit Categories
We have categorized commits into the following types:

- **feat**: A new feature or enhancement.
- **fix**: A bug fix.
- **docs**: Documentation changes.
- **style**: Code style changes (e.g., formatting).
- **refactor**: Code refactoring without new features or bug fixes.
- **test**: Adding or modifying tests.
- **chore**: Maintenance tasks, tooling, or other non-feature related changes.

### Commit Message Format
- Start with the commit category in lowercase.
- Use a colon and a space after the category.
- Limit the subject line to 50 characters.
- Use the imperative mood (e.g., "feat: Add data preprocessing").
- Optionally, add an extended description separated by a blank line.

### Examples
```plaintext
feat: Implement logistic regression model

This commit implements a logistic regression model to classify weather events into three categories: 'regular', 'cyclone', and 'atmospheric river'.
```

```plaintext 
fix: Resolve issue with data loading

Fixes a bug where the data loading function was failing due to incorrect file paths.
```

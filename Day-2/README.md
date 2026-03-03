# Golden Rules Diagnostics breif of what happened

## What is this image about?
This image shows the **golden rules** for diagnosing problems in deep learning models. It helps us understand what's going wrong when our model doesn't work well.

## The Main Rules

**Rule 1: Check Your Data**
- Make sure your data is good and clean
- Look for missing or broken data
- Check if the data is balanced

**Rule 2: Start Simple**
- Begin with a simple model first
- Don't make it too complicated right away
- Make sure the simple model works

**Rule 3: Check if Model is Learning**
- Look at the training loss
- Does it go down over time?
- If not, something is wrong with training

**Rule 4: Check for Overfitting**
- Compare training loss and validation loss
- If training loss is much lower, your model might be overfitting
- This means it only memorized the data, not learning

**Rule 5: Adjust Learning Rate**
- If loss doesn't go down, try changing the learning rate
- Sometimes it needs to be bigger or smaller
- This helps the model learn better

**Rule 6: More Data = Better**
- More training data usually helps the model learn better
- It helps prevent overfitting too
- Try to collect more data if you can

## Why is this important?
When we build deep learning models, many things can go wrong. These golden rules help us find the problems step by step and fix them. It's like a guide for debugging our models!

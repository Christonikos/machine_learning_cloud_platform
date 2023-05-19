## Project: A Dynamic Risk Assessment System ##
In this project I built and monitored an ML model to predict attrition risk.

### Project goals  ###
A system that achieves the following:
  * Data ingestion: Automatically checks a database for new data that can be used for model training. Compiles all training data to a training dataset and saves it to persistent storage.Writes metrics related to the completed data ingestion tasks to persistent storage.

  * Training, scoring, and deploying: Writes scripts that trains an ML model that predicts attrition risk, and scores the model. Writes the model and the scoring metrics to persistent storage.

  * Diagnostics: Determines and saves summary statistics related to a dataset. Times the performance of model training and scoring scripts. Checks for dependency changes and package updates.

  * Reporting: Automatically generates plots and documents that report on model metrics. Provides an API endpoint that can return model predictions and metrics.

  * Process Automation: Creates a script and cron job that automatically run all previous steps at regular intervals.

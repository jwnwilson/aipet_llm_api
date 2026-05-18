#TODO

# Consolidate to llm-api
- Rename this project to "llm-api" and make it a generic training platform. Remove "aipet" references
- Add llm-ui to this project

## LLM training pipeline
- Better error handling on the workflows to update runs to "error" with "error_msg" shown on llm-ui
- Run overrides are not flowing to the pipeline investigate add tests then fix that
- Allow user to upload training and eval data via ui
- Select a model via ui
- Select a platform to train on via ui
- Improve eval for the training and make that data available via API
- Report the eval for the model in a UI

## Better llm-api architecture
- Make the llm API load instantly by decoupling it from model loading.
- Setup functionality to set a model to "active"
- Spin up a container for each "active" llm models requesting the right memory for the model.
- Setup scaling for each model independanlty
    - Scale to 0 if model is not used for 1 hour
- Track status on active llm models to show on the ui.
- Handle requests to loading models and return a result or good http status with "not_ready_yet" 

## LLM API
- Expose inference for each model via API tab on UI
- Provide an apikey for a user to run inference on their model
- Add rate limiting per user 

## Fast E2E tests
- Re-enable E2E tests on CI/CD to run once a day or something


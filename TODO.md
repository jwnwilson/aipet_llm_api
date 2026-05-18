#TODO

## LLM training platform
- Rename this project to "llm-api" and make it a generic training platform. Remove "aipet" references
- Add llm-ui to this project
- Better error handling on the workflows to update runs to "error" with "error_msg" shown on llm-uo
- Allow user to upload training and eval data
- Select a model
- Select a platform to train on
- Setup eval for the training
- Report the eval for the model in a UI
- Make successful models available to test via the AP

## Better llm-api architecture
- Make the llm API load instantly and decouple models from the main api.
- Spin up a container for each "active" llm models requesting the right memory for the model.
- Setup scaling for each model independanlty
- Track status on active llm models to show on the ui.
- Handle requests to loading models and return a result or good http status with "not_ready_yet" 

## Fast E2E tests
- Re-enable E2E tests on CI/CD to run once a day or something


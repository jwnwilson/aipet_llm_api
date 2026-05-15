#TODO

Setup plan and execute the following in parallel:
1. Setup Authentication
    - Validate with tests
2. Deploy this service to kubernetes cluster
    - Validate with tests
    - Add dns setting for aipet-llm api

Next Features

## Fast E2E tests
- Re-enable E2E tests on CI/CD to run once a day or something

## LLM training platform
- Allow user to upload training and eval data
- Select a model
- Select a platform to train on
- Setup eval for the training
- Report the eval for the model in a UI
- Make successful models available to test via the AP
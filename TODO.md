#TODO

1. Setup E2E Tests for all compute E2E to inference
2. Setup tests with valid presaved model in S3 
3. Fix all tests setup proper ci cd tests

Setup plan and execute the following in parallel:
1. Setup Authentication
    - Validate with tests
2. Deploy this service to kubernetes cluster
    - Validate with tests
3. Make successful models available to test via the API

Product ideas

## LLM training platform
- Allow user to upload training and eval data
- Setup eval for the training
- Select a model
- Select a platform to train on
- Report the eval for the model
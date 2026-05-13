#TODO

1. Setup working vastai with end to end test
    - Training working
    - Validate export 
2. Setup working runpod with end to end test
    - Training working
    - Validate export 
3. Setup E2E Tests for all compute E2E to inference
4. Setup tests with valid presaved model in S3 
5. Fix all tests setup proper ci cd tests

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
#TODO

1. Validate llm ui works with new API
    - This looks broken, add integration tests to ensure this doesn't break in future
    - Add ability to export / evaluate / set current model for the api in the ui
2. Setup working vastai with end to end test
3. Setup end to end test with valid presaved model in S3 or something 

Setup plan and execute the following in parallel:
1. Setup Authentication
2. Deploy this service to kubernetes cluster
3. Validate colabs adapter with project
    - Setup GCP storage adapter for files here.
4. Make successful models available to test via the API
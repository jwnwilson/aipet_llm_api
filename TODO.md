#TODO

1. Verify new large model works well with the AIPET game
2. Verify ability to re eval and re export with async functionality
3. Validate llm ui works with new API
    - This looks broken, add integration tests to ensure this doesn't break in future
    - Add ability to export / evaluate / set current model for the api in the ui

Setup plan and execute the following in parallel:
4. Deploy this service to kubernetes cluster
5. Validate colabs adapter with project
    - Setup GCP storage adapter for files here.
5. Make successful models available to test via the API
# Product Requirements Document

---

## Problem

I want to build an API which can run a lightweight LLM able to run on a raspberry pii. The LLMs task will be to power an AI pet game, it will take in simplified 3D scene data and also the pet's stats like: hunger, boredom, social, toilet, tiredness. The response will return a desired "action" which should be one of a few presets and object in the scene it wants to interfact with. This will drive a ai pet character in a 3D scene.

As the product develops I want to develop the LLM into an AIpet agent which will be given access to tools / skills to allow it to query the scene for the closest object it wants to interact with.

## Vision

We have an AIpet LLM running on an API that powers interesting behaviour for a 3D browser game. The behaviour has grown complex enough to similate the behaviour of a digital pet that can interact with player avatars.

## Core Features

<!-- What must the product do? Use MoSCoW: Must / Should / Could / Won't -->

### Must have
- Consistent schema for scene data, pet needs and the pet response that can be used by an AIPet game to power an AI pet.
- Be able to train a small LLM that can run efficiently on a Raspberry Pi 5 (8GB)
- Have a FastAPI interface for the model
- Be able to parse scene data (object type + distance to pet) and pet stats, and return a valid action and optional target object
- Supported actions with required target object types:

  | Action    | Target object required | Valid target types  |
  |-----------|----------------------|---------------------|
  | EAT       | Yes                  | bowl                |
  | DRINK     | Yes                  | bowl                |
  | PLAY      | Yes                  | toy                 |
  | FETCH     | Yes                  | toy                 |
  | SLEEP     | Yes                  | bed                 |
  | SOCIAL    | Yes                  | player, pet         |
  | FOLLOW    | Yes                  | player, pet         |
  | TOILET    | No                   | —                   |
  | IDLE      | No                   | —                   |
  | EXPLORE   | No                   | —                   |

- Only actions whose target object type is present in the scene are valid choices — this constrains the LLM output space
- Scene objects: fixed (`bowl`, `bed`, `toy`) and dynamic (`player`, `pet`); represented as `{type, id, distance}` — no position coordinates
- Latency budget: inference every few seconds (pet completes an action before the next decision)

### Should have
- Consistent architecture for the project
- Hexagonal architecture with the LLM domain logic re-usable
- No coupled API and domain logic

### Could have (post-v1)
- Pet personality traits (e.g. lazy, social, playful) that bias action selection

### Won't have (v1)
- agententic functionality

## User Stories

- As an AIPet user, I want to be able to send scene data and pet stats to an llm so that I can get a valid response with action and optional scene object to interact with which controls my 3D AIPet game.
- As an AIPet user, I want to be able to pass data and recieve a response from an API with a consistent schema.

## Success Metrics

| Metric | Target |
|--------|--------|
| % of valid responses matching schema generated from the llm     |  > 95%      |

## Technical Constraints

- Python 3.12+, FastAPI, pytest
- Inference runtime: llama-cpp-python with GGUF quantised model (no GPU, ARM64)
- Target hardware: Raspberry Pi 5 (8GB)
- Deployment: Docker container (Kubernetes setup is out of scope for v1)
- Browser game is being built in parallel — API schema must be agreed jointly before both sides start
- Model size: up to ~3B parameters viable given RPi 5 8GB + few-second latency budget

## Open Questions

<!-- Things still to decide. Use AI to work through these. -->

1. ?

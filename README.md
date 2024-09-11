# Inner Dialogue: Recursive Self-Reflection for LLMs

**Inner Dialogue** is an experimental project that enables recursive self-reflection within a Large Language Model (LLM). The model can call itself to refine its responses based on prior output, allowing it to "think harder" about problems and simulate an internal dialogue. This process helps the model break down complex questions or reattempt questions when its first response isn't adequate, enhancing accuracy and depth.

## ToDo
* The python interpreter tool is unconfined so far. We need to build the interpreter into a docker container or else the LLM might accidentally destroy stuff
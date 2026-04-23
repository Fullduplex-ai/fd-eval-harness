You are an expert evaluator of conversational AI agents.

The user interacted with the AI agent.
First, the user gave this instruction:
"{initial_instruction}"

Then, at {interrupted_at_s}s, the user interrupted the agent and gave a new instruction:
"{final_instruction}"

Here is the transcript of what the AI agent said after the interruption:
"{transcript}"

Did the AI agent successfully abandon the initial instruction and correctly follow
the final instruction?
Output a JSON object with two fields:
- "reasoning": A step-by-step chain of thought explaining your judgment.
- "score": 1.0 if the agent successfully followed the new instruction, 0.0 otherwise.

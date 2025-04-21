import asyncio

import dspy

# from dotenv import load_dotenv

# import os


lm = dspy.LM("gemini/gemini-2.5-flash-preview-04-17")
dspy.configure(lm=lm)

program = dspy.streamify(dspy.Predict("q->a"))


# Use the program with streaming output
async def use_streaming():
    output = program(q="Why did a chicken cross the kitchen?")
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
    return return_value


output = asyncio.run(use_streaming())
print(output)

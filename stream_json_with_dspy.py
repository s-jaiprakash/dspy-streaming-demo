import dspy
import ujson
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/stream_json.html")

@app.get("/stream_json") # Redundant endpoint, you can remove this or make it point to root
async def stream_json():
    return FileResponse("static/stream_json.html")

lm = dspy.LM('gemini/gemini-2.5-flash-preview-04-17')
dspy.configure(lm=lm)


class Relevance(dspy.Signature):
    is_it_relevant: bool = dspy.OutputField(desc="the given problem statement relevant to the pharma industry")
    justification: str = dspy.OutputField(desc="justify why it's relevant or why it's not relevant")


# Define PharmaProblemValidator Signature with structured OutputField using Dict
class PharmaProblemValidator(dspy.Signature):
    """Analyze a pharma problem statement and validate it against key factors.
    Output is a JSON-like dictionary structure.
    """
    context: str = dspy.InputField(desc="facts here are assumed to be true")
    problem_statement = dspy.InputField(desc="The problem statement to validate")
    relevance: Relevance = dspy.OutputField(desc="does problem statement relevant to the pharma or not")
    is_personas_mentioned: bool = dspy.OutputField(desc="does problem statement mentioned any personas")
    relevance_justification_personas: str = dspy.OutputField(desc="give justification how it's related to pharma or why it's not")

context = """

You are a problem evaluator, please help evaluate the below problem statements against given metrics like relevance to pharma and personas mentioned
"""
PredictPharmaValidation = dspy.Predict(PharmaProblemValidator)
stream_program = dspy.streamify(PredictPharmaValidation)

# Store streams in a dictionary, keyed by a simple counter (not production-ready)
stream_registry = {}
stream_counter = 0

class QuestionInput(BaseModel):
    problem_statement: str

@app.post("/ask")
async def ask_question_post(data: QuestionInput): # Renamed to avoid conflict
    global stream_counter
    stream_id = f"stream_{stream_counter}"
    stream_counter += 1
    # We are now expecting a dictionary as output from stream_program, not structured classes
    stream_generator = stream_program(context=context, problem_statement=data.problem_statement)
    stream_registry[stream_id] = stream_generator
    return {"stream_url": f"/stream/{stream_id}"} # Return stream URL

@app.get("/stream/{stream_id}") # Added /stream endpoint
async def stream_output(stream_id: str):
    stream_generator = stream_registry.get(stream_id)
    if stream_generator is None:
        raise HTTPException(status_code=404, detail="Stream not found")

    async def streamer():
        async for chunk in stream_generator: # Use async for to iterate over async generator
            # Convert chunk to dict if it's a DSPy Signature object
            if isinstance(chunk, (Relevance, )): # Add other Signature classes if needed
                chunk_dict = {}
                for field_name in chunk.__class__.__fields__:
                    chunk_dict[field_name] = getattr(chunk, field_name)
                yield f"data: {ujson.dumps(chunk_dict)}\n\n"
            else:
                # If it's not a Signature, try to json serialize directly, assuming it's meant to be json
                try:
                    yield f"data: {ujson.dumps(chunk)}\n\n"
                except TypeError: # Handle cases where chunk is not directly json serializable
                    yield f"data: {str(chunk)}\n\n" # Fallback to string representation
    return StreamingResponse(streamer(), media_type="text/event-stream")
from typing import List, Optional

import dspy
import ujson
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from litellm import ModelResponseStream
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
    is_it_relevant: bool = dspy.OutputField(desc="Is the problem statement relevant to the pharma industry?")
    justification: str = dspy.OutputField(desc="Justify why it's relevant or why it's not relevant to pharma.")


class PersonasValidation(dspy.Signature):
    """Validate if personas are mentioned in the problem statement."""
    is_personas_mentioned: bool = dspy.OutputField(desc="Does the problem statement mention any personas from the pharma industry?")
    mentioned_personas: Optional[List[str]] = dspy.OutputField(desc="List the mentioned personas if any, otherwise leave empty. Personas are: Clinical Data Scientist, Regulatory and Compliance Specialist, Pharmacovigilance and Safety Officer, R&D and Bioinformatics Scientist, Medical Affairs and Market Access Specialist, Sales and Key Account Manager, Marketing and Commercial Strategy Manager, Business Development and Partnerships Lead, IT and Data Engineer, C-Suite Executive", allow_empty=True)
    personas_validation_output: str = dspy.OutputField(desc="Explain if personas are mentioned, which ones, and justify the impact or if not mentioned justify why it's necessary.")

class UrgencyValidation(dspy.Signature):
    """Classify and validate the urgency of the problem."""
    urgency_level: Optional[str] = dspy.OutputField(desc="Classify the urgency level as Low, Medium, or High based on the problem statement's impact. If no urgency can be determined, output null", allow_empty=True)
    urgency_validation_output: str = dspy.OutputField(desc="Explain the urgency level and justify based on impact on pharma operations, compliance, or business risk. If no urgency mentioned, explain why it's important to define urgency.")

class CurrentProcessValidation(dspy.Signature):
    """Verify and describe the current process mentioned in the problem."""
    is_process_mentioned: bool = dspy.OutputField(desc="Does the problem statement describe the current process for handling the issue?")
    current_process_description: Optional[str] = dspy.OutputField(desc="Describe the current process as mentioned in the problem statement. If no process is mentioned, output null", allow_empty=True)
    process_validation_output: str = dspy.OutputField(desc="Explain the current process described, highlight inefficiencies or limitations if mentioned. If no process is described, explain why describing the current process is important.")

class ManHoursValidation(dspy.Signature):
    """Validate and categorize the man-hours mentioned in the problem."""
    are_man_hours_mentioned: bool = dspy.OutputField(desc="Does the problem statement quantify man-hours spent on the problem?")
    estimated_man_hours_range: Optional[str] = dspy.OutputField(desc="Map the mentioned man-hours to predefined pharma-relevant ranges: 1-5 hours per week (Minor inefficiency), 5-10 hours per week (Noticeable resource drain), 10-20 hours per week (Major productivity loss), 20+ hours per week (Severe bottleneck). If no man-hours mentioned, output null", allow_empty=True)
    man_hours_validation_output: str = dspy.OutputField(desc="Explain the operational or financial impact based on the man-hours range. If man-hours are not quantified, explain why quantifying man-hours is important for business case justification.")


# Define PharmaProblemValidator Signature with structured OutputField using Dict
class DetailedPharmaProblemValidator(dspy.Signature):
    """Analyze a pharma problem statement and validate it against key factors.
    Output is a JSON-like dictionary structure.
    """
    context: str = dspy.InputField(desc="facts here are assumed to be true")
    problem_statement = dspy.InputField(desc="The problem statement to validate")
    relevance: Relevance = dspy.OutputField(desc="Assess the relevance of the problem statement to the pharma industry.")
    personas_validation: PersonasValidation = dspy.OutputField(desc="Validate if personas are mentioned and their relevance.")
    urgency_validation: UrgencyValidation = dspy.OutputField(desc="Validate and classify the urgency of the problem.")
    current_process_validation: CurrentProcessValidation = dspy.OutputField(desc="Verify and describe the current process.")
    man_hours_validation: ManHoursValidation = dspy.OutputField(desc="Validate and categorize the man-hours mentioned.")


context = """
You are an AI-driven Pharma Domain Expert specializing in validating problem statements related to inefficiencies in the pharmaceutical industry. Your goal is to analyze the given problem statement and validate it against four critical factors: Personas, Urgency, Current Process, and Man-Hours.

For Personas Validation:
Check if the problem statement mentions affected roles from this list:
- Clinical Data Scientist
- Regulatory and Compliance Specialist
- Pharmacovigilance and Safety Officer
- R&D and Bioinformatics Scientist
- Medical Affairs and Market Access Specialist
- Sales and Key Account Manager
- Marketing and Commercial Strategy Manager
- Business Development and Partnerships Lead
- IT and Data Engineer
- C-Suite Executive

Validation Output for Personas:
- If personas are mentioned: "The problem statement identifies [Mentioned Personas], which are directly affected due to [Explain impact]. This ensures that the right stakeholders are considered."
- If missing: "The problem statement does not specify affected personas. This is necessary to understand who experiences the problem and who will benefit from the solution."


For Urgency Validation:
Classify urgency based on the problem statement’s impact:
- Low – Minor inefficiencies but no critical business disruption
- Medium – Workflow issues affecting timelines, compliance, or efficiency
- High – Regulatory risk, financial loss, or significant delays in drug development

Validation Output for Urgency:
- If urgency is defined: "The urgency level is [Low, Medium, High] because [Explain the impact on pharma operations, compliance, or business risk]."
- If missing: "The problem statement does not define urgency, making it difficult to prioritize resource allocation and problem resolution."


For Current Process Validation:
Verify how the issue is currently managed:
- Manual processes such as spreadsheets or human intervention
- Rule-based automation that is partially automated but inefficient
- No clear process leading to data or compliance gaps

Validation Output for Current Process:
- If process is mentioned: "The problem statement explains the current process as [Describe Current Process], highlighting [Inefficiencies or limitations]. This helps compare current methods with potential improvements."
- If missing: "The problem statement does not describe the current process, making it harder to evaluate the necessity of change."


For Current Man-Hours Validation:
Convert mentioned hours into predefined pharma-relevant ranges:
- 1-5 hours per week – Minor inefficiency
- 5-10 hours per week – Noticeable resource drain
- 10-20 hours per week – Major productivity loss
- 20+ hours per week – Severe bottleneck

Validation Output for Man-Hours:
- If man-hours are mentioned: "The problem statement estimates [X hours], mapped to [Mapped Range], indicating [Explain the operational or financial impact]."
- If missing: "The problem statement does not quantify man-hours, making it harder to justify the business case."

You are to provide output in JSON format, structured as defined in the `DetailedPharmaProblemValidator` signature, streaming each section of the validation as it becomes available.
"""
PredictDetailedPharmaValidation = dspy.Predict(DetailedPharmaProblemValidator)
stream_program = dspy.streamify(PredictDetailedPharmaValidation)

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


@app.get("/stream/{stream_id}")
async def stream_output(stream_id: str):
    stream_generator = stream_registry.get(stream_id)
    if stream_generator is None:
        raise HTTPException(status_code=404, detail="Stream not found")


    async def streamer():
        async for chunk in stream_generator:
            if isinstance(chunk, (
                Relevance,
                PersonasValidation,
                UrgencyValidation,
                CurrentProcessValidation,
                ManHoursValidation,
                DetailedPharmaProblemValidator
            )):
                yield f"data: {ujson.dumps(chunk.model_dump())}\n\n"

            elif isinstance(chunk, dspy.Prediction):
                yield f"data: {ujson.dumps(chunk.toDict())}\n\n"

            elif isinstance(chunk, ModelResponseStream):
                # serialize properly using model_dump (for Pydantic V2)
                yield f"data: {ujson.dumps(chunk.model_dump(exclude_unset=True))}\n\n"

            else:
                try:
                    yield f"data: {ujson.dumps(chunk)}\n\n"
                except TypeError:
                    yield f"data: {str(chunk)}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")

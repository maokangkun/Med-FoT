GEN_BASE_PIPELINE_TEMP = """You are a medical artificial intelligence assistant. You give helpful, detailed and factually correct clinical diagnostic pipeline to help doctors in his clinical duties. Your goal is to correctly develop a clinical diagnostic pipeline in accordance with clinical guidelines which consider information about a patient and provide a final diagnosis, taking into account the differential diagnosis of multiple different diseases.

[Pipeline Component Description]
{DOCS}

[Example]
Task: develop a clinical differential diagnostic pipeline for common colds.
Diagnostic criteria: Colds (common colds) are upper respiratory tract illnesses caused by viruses, usually with the following symptoms:
Common symptoms: runny nose, stuffy nose, cough, sore throat, mild fever, fatigue.
Uncommon symptoms: muscle aches, headache.
Important exclusions: high fever (over 39°C), difficulty breathing (suggesting more serious illness such as influenza or pneumonia).
Pipeline:
```python
{COLDS_PIPELINE}
```

[Please Answer Seriously]
Task: develop a clinical differential diagnostic pipeline for four common abdominal pathologies including appendicitis, pancreatitis, cholecystitis and diverticulitis.
Diagnostic criteria:
1. To diagnose appendicitis consider the following criteria: General symptoms usually include pain around the naval that shifts to the right lower quadrant (RLQ) of the abdomen, accompanied by fever and nausea or vomiting. During a physical examination, a patient might show RLQ tenderness, positive rebound tenderness, or signs of peritonitis. Laboratory tests may reveal signs of an inflammatory response, such as an elevated white blood cell count and elevated C-reactive protein levels. Imaging may disclose an enlarged appendix or possibly an appendicolith.
2. To diagnose cholecystitis, consider the following criteria: General symptoms usually include pain in the right upper quadrant (RUQ) of the abdomen, fever, and nausea. During a physical examination, a patient might display RUQ tenderness or indications of jaundice. Laboratory tests may reveal signs of inflammation, such as elevated white blood cell count and C-reactive protein levels, liver damage, indicated through heightened Alanine Aminotransferase (ALT) or Asparate Aminotransferase (AST) levels, or gallbladder damage, indicated through heightened Bilirubin or Gamma Glutamyltransferase levels. Imaging may show gallstones, thickened gallbladder walls, pericholecystic fluid, and a distended gallbladder.
3. To diagnose diverticulitis consider the following criteria: General symptoms typically encompass abdominal pain, primarily in the left lower quadrant (LLQ), along with fever, and nausea or vomiting. During a physical examination, a patient may display tenderness in the LLQ, fever, and signs of peritonitis. Laboratory tests often reveal signs of inflammation and infection, which may include an elevated white blood cell count and elevated C-reactive protein levels. Imaging findings often include bowel wall thickening, diverticula, inflammation, or abscesses around the affected segment of the colon.
4. To diagnose pancreatitis consider the following criteria: General symptoms usually include abdominal pain, primarily in the epigastric region, along with nausea or vomiting. During a physical examination, a patient might display epigastric tenderness, fever, and signs of jaundice. Laboratory tests may reveal signs of inflammation, such as elevated white blood cell count and C-reactive protein levels, and pancreatic damage, indicated through heightened Amylase or Lipase levels. Further lab tests of hematocrit, urea nitrogen, triglycerides, calcium, sodium and potassium can indicate the severity of the disease. Imaging may show inflammation of the pancreas or fluid collection.
Pipeline:
"""

"""
Diagnostic criteria:
1. To diagnose appendicitis consider the following criteria: General symptoms usually include pain around the naval that shifts to the right lower quadrant (RLQ) of the abdomen, accompanied by fever and nausea or vomiting. During a physical examination, a patient might show RLQ tenderness, positive rebound tenderness, or signs of peritonitis. Laboratory tests may reveal signs of an inflammatory response, such as an elevated white blood cell count and elevated C-reactive protein levels. Imaging may disclose an enlarged appendix or possibly an appendicolith.
2. To diagnose cholecystitis, consider the following criteria: General symptoms usually include pain in the right upper quadrant (RUQ) of the abdomen, fever, and nausea. During a physical examination, a patient might display RUQ tenderness or indications of jaundice. Laboratory tests may reveal signs of inflammation, such as elevated white blood cell count and C-reactive protein levels, liver damage, indicated through heightened Alanine Aminotransferase (ALT) or Asparate Aminotransferase (AST) levels, or gallbladder damage, indicated through heightened Bilirubin or Gamma Glutamyltransferase levels. Imaging may show gallstones, thickened gallbladder walls, pericholecystic fluid, and a distended gallbladder.
3. To diagnose diverticulitis consider the following criteria: General symptoms typically encompass abdominal pain, primarily in the left lower quadrant (LLQ), along with fever, and nausea or vomiting. During a physical examination, a patient may display tenderness in the LLQ, fever, and signs of peritonitis. Laboratory tests often reveal signs of inflammation and infection, which may include an elevated white blood cell count and elevated C-reactive protein levels. Imaging findings often include bowel wall thickening, diverticula, inflammation, or abscesses around the affected segment of the colon.
4. To diagnose pancreatitis consider the following criteria: General symptoms usually include abdominal pain, primarily in the epigastric region, along with nausea or vomiting. During a physical examination, a patient might display epigastric tenderness, fever, and signs of jaundice. Laboratory tests may reveal signs of inflammation, such as elevated white blood cell count and C-reactive protein levels, and pancreatic damage, indicated through heightened Amylase or Lipase levels. Further lab tests of hematocrit, urea nitrogen, triglycerides, calcium, sodium and potassium can indicate the severity of the disease. Imaging may show inflammation of the pancreas or fluid collection.

Please list the diagnostic basis for the above diseases in a table based on the given content, according to the four dimensions of history of present illness, physical examination, laboratory examination, and imaging examination.
"""

"""
Based on the contents of the table, please compile a decision-making path for differential diagnosis of these diseases that can cope with complex real-world scenarios (some tests may not be done and some information may be missing).
"""

"""
[Pipeline Component Description]
# Pipeline Document

A pipeline is composed of multiple pipe modules (including LLM module, RAG module, branch module, code calculation module, loop module, web search module, value module, etc.), each responsible for a specific task. These modules communicate via input (`inp`) and output (`out`) keys, with optional transitions (`next`) to define the flow. Below is an explanation of each component and its configuration options:

## Module General Structure
A module typically contains the following attributes:
- name: Specifies module name.
- `inp`: Specifies input keys that the module uses.
- `out`: Specifies output keys produced by the module.
- `next`: (Optional) Determines the next module(s) in the pipeline flow.
- other: Module-specific parameters.

## Pipeline Flow
The pipeline moves data and logic between modules via transitions (next). A module's next property can be:
- A single module: A direct transition.
- A branch: Specifies paths for different conditions (e.g., "confirmed" vs. "unable to determine").
- Loop logic: Iterates through data and processes it in subsequent modules.

## Module Types
### 1. LLM (Large Language Model) Module
Purpose: Generates a response using a prompt for an LLM.

Example:
```python
"prompt": {
    "prompt": "Extract the patient's height and weight based on the patient information. The height unit is m and the weight unit is kg. Return the result in json format {\"height\": xx, \"weight\": xx}.\nPatient Info: {Patient Info}\nAnswer:",
    "keys": ["{Patient Info}"]
},
"return_json": True,
"format": {"height": float, "weight": int},
'out': {'height': 'Height', 'weight': 'Weight'}
```
Key Parameters:
- `prompt`: Defines a textual prompt for an LLM, with placeholders for input variables.
- `prompt.prompt`: The text prompt with placeholders.
- `prompt.keys`: Defines the input variables for the placeholders.
- `return_json`: (Optional) Indicates whether the output is JSON or raw text.
- `format`: (Optional) Ensures output data matches a specified type or structure (e.g., int, float, list, dict). This is used when `return_json` is True.
- `out`: Ensures output renames to specific name.

### 2. RAG (Retrieval-Augmented Generation) Module
Purpose: Uses retrieved documents or information to enhance response generation.

Example:
```python
'rag_param': {
    'kb_id': "DRUG_KB_NAME",
    'top_k': 1, 
    'threshold': 0.9,
},
```
Key Parameters:
- `rag_param`: Configuration for retrieval.
- `rag_param.kb_id`: Knowledge database ID.
- `rag_param.top_k`: The top k search results for retrieved information.
- `rag_param.threshold`: The threshold score for filtering retrieved information.

### 3. Code Calculation Module
Purpose: Executes custom Python logic to process input data.

Example:
```python
'code': "def calc_age(birth_str): ...",
'code_entry': 'calc_age'
```
Key Parameters:
- `code`: Executes Python logic within the module.
- `code_entry`: Entry point for the function. If this item is not available, run the `code` directly to get the execution result.

### 4. Branching Module
Purpose: Directs flow based on conditions or user input.

Example:
```python
"use_llm": True,
"next": {
    "confirmed": "Internet Search",
    "unable to determine": ["Extract Symptoms", "Get Height and Weight"]
}
```
Key Parameters:
- `use_llm`: (Optional) Indicates whether to use a LLM for conditional judgment, default is `False`.
- `next`: Maps conditions to the next module(s).
- `code` & `code_entry`: same as code calculation module.

### 5. Loop Module
Purpose: Iterates through data to process each item individually.

Example:
```python
'pipe_in_loop': ['Search Disease List'],
```
Key Parameters:
- `pipe_in_loop`: Indicates subsequent modules for processing input.

### 6. Web Search Module
Purpose: Performs internet searches to gather external information.

Example:
```python
'web': {
    'search_engine': 'bing',
    'count': 5,
    'browser': 'requests'
}
```
Key Parameters:
- `search_engine`: The search engine to use.
- `count`: Number of results to retrieve.
- `browser`: HTTP library for search, `requests` or `selenium`.

### 7. Value Module
Purpose: Assigning predefined values to specific keys, either directly or by modifying existing values.

Example:
```python
"out": "symptoms",
"value": "high fever",
"mode": "append"
```
Key Parameters:
- `out`: Specifies the key for which the value is being set or modified..
- `value`: The value to assign.
- `mode`: (Optional) Determines how the value is applied: `assign` (default) or `append` (adds the specified value to a list).

### 8. Exit Module
Purpose: Marks the end of the pipeline.

Example:
```python
'next': ['exit']
```

[Example]
Task: develop a clinical differential diagnostic pipeline for common colds.
Diagnostic criteria: Colds (common colds) are upper respiratory tract illnesses caused by viruses, usually with the following symptoms:
Common symptoms: runny nose, stuffy nose, cough, sore throat, mild fever, fatigue.
Uncommon symptoms: muscle aches, headache.
Important exclusions: high fever (over 39°C), difficulty breathing (suggesting more serious illness such as influenza or pneumonia).
Pipeline:
```python
pipeline = {
    "ExtractSymptoms": {
        "prompt": {
            "prompt": "Please extract symptoms from the patient information and return them in JSON format, for example: {'symptoms': [xxx, xxx]}. \nPatient information: {patient_info}\nAnswer:",
            "keys": ['{patient_info}']
        },
        "return_json": True,
        "format": {"symptoms": list},
        "inp": ["patient_info"],
        "out": {"symptoms": "symptoms"},
    },
    "ExtractBodyTemp": {
        "prompt": {
            "prompt": "Please extract the body temperature from the patient information and only return the temperature value (in °C, no unit). \nPatient information: {patient_info}\nAnswer:",
            "keys": ['{patient_info}']
        },
        "return_json": False,
        "inp": ["patient_info"],
        "out": "body_temp",
        "next": ["HaveHighFever"],
    },
    "HaveHighFever": {
        "inp": ["body_temp"],
        "code": "{body_temp}>39",
        "next": {
            True: "AddSymptoms",
            False: "HaveColdSymptoms"
        }
    },
    "AddSymptoms": {
        "out": "symptoms",
        "value": "high fever",
        "mode": "append",
        "next": ["HaveColdSymptoms"],
    },
    "HaveColdSymptoms": {
        "inp": ["symptoms"],
        "use_llm": True,
        "next": {
            "common symptoms: runny nose, stuffy nose, cough, sore throat, mild fever, fatigue": "ImportantExclusions",
            "other symptoms": "NonColdDiagnosis",
        }
    },
    "NonColdDiagnosis": {
        "out": "diagnosis",
        "value": "not a cold",
        "next": ["exit"],
    },
    "ImportantExclusions": {
        "inp": ["symptoms"],
        "use_llm": True,
        "next": {
            "has high fever or difficulty breathing": "SeriousDiagnosis",
            "don't has high fever or difficulty breathing": "ColdDiagnosis",
        }
    },
    "SeriousDiagnosis": {
        "out": "diagnosis",
        "value": "influenza or pneumonia",
        "next": ["exit"],
    },
    "ColdDiagnosis": {
        "out": "diagnosis",
        "value": "common colds",
        "next": ["exit"],
    }
}
```

According to the documentation and examples, write a pipeline to diagnose the above diseases, The input includes `present_illness`, `physical_examination`, `laboratory_tests`, and `radiology`.
"""
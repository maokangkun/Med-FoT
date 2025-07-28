pipeline = {
    "GetDiagnosticPoints": {
        "prompt": {
            "prompt": "Based on the diagnostic criteria of these diseases {candidate_diseases}, list key diagnostic points for differential diagnosis from four perspectives: present illness, physical examination, laboratory tests, and radiology. Return a structured answer in JSON format, e.g.\n```json\n{\"present_illness_diag_points\": [...], \"physical_exam_diag_points\": [...], \"lab_tests_diag_points\": [...], \"radiology_diag_points\": [...]}\n```\n\nAnswer:",
            "keys": ['{candidate_diseases}']
        },
        "retry": 10,
        "return_json": True,
        "format": {"present_illness_diag_points": list, "physical_exam_diag_points": list, "lab_tests_diag_points": list, "radiology_diag_points": list},
        "inp": ["candidate_diseases"],
        "out": {
            "present_illness_diag_points": "phi_diag_points",
            "physical_exam_diag_points": "pe_diag_points",
            "lab_tests_diag_points": "lab_diag_points",
            "radiology_diag_points": "rad_diag_points"
        },
        "next": ["ExtractPHIFindings", "ExtractPEFindings", "ExtractLabFindings", "ExtractRadFindings"],
    },
    "ExtractPHIFindings": {
        "prompt": {
            "prompt": "Extract the relevant present illness findings based on the key diagnostic points for differential diagnosis. Return a structured answer in JSON format, e.g.\n```json\n{\"findings\": [\"xxx\", \"xxx\"]}\n```\n\nPresent illness: {present_illness}\nKey diagnostic points: {key_diag_points}\nAnswer:",
            "keys": ["{present_illness}", "{key_diag_points}"]
        },
        "return_json": True,
        "format": {"findings": list},
        "inp": ["present_illness", "phi_diag_points"],
        "out": {"findings": "phi_findings"}
    },
    "ExtractPEFindings": {
        "prompt": {
            "prompt": "Extract the relevant physical examination findings based on the key diagnostic points for differential diagnosis. Return a structured answer in JSON format, e.g.\n```json\n{\"findings\": [\"xxx\", \"xxx\"]}\n```\n\nPhysical examination: {physical_examination}\nKey diagnostic points: {key_diag_points}\nAnswer:",
            "keys": ["{physical_examination}", "{key_diag_points}"]
        },
        "return_json": True,
        "format": {"findings": list},
        "inp": ["physical_examination", "pe_diag_points"],
        "out": {"findings": "pe_findings"}
    },
    "ExtractLabFindings": {
        "prompt": {
            "prompt": "Extract the relevant laboratory findings based on the key diagnostic points for differential diagnosis. Return a structured answer in JSON format, e.g.\n```json\n{\"findings\": [\"xxx\", \"xxx\"]}\n```\n\nLaboratory tests: {laboratory_tests}\nKey diagnostic points: {key_diag_points}\nAnswer:",
            "keys": ["{laboratory_tests}", "{key_diag_points}"]
        },
        "return_json": True,
        "format": {"findings": list},
        "inp": ["laboratory_tests", "lab_diag_points"],
        "out": {"findings": "lab_findings"}
    },
    "ExtractRadFindings": {
        "prompt": {
            "prompt": "Extract the relevant imaging findings based on the key diagnostic points for differential diagnosis. Return a structured answer in JSON format, e.g.\n```json\n{\"findings\": [\"xxx\", \"xxx\"]}\n```\n\nRadiology: {radiology}\nKey diagnostic points: {key_diag_points}\nAnswer:",
            "keys": ["{radiology}", "{key_diag_points}"]
        },
        "return_json": True,
        "format": {"findings": list},
        "inp": ["radiology", "rad_diag_points"],
        "out": {"findings": "rad_findings"}
    },
    "PreliminaryDiagnosis": {
        "prompt": {
            "prompt": "Make a preliminary diagnosis based on the patient's present illness and physical examination, return one or more possible items from {candidate_diseases} and the corresponding reason and confidence in JSON format, e.g.\n```json\n{\"preliminary_diagnosis\": [{\"reason\": \"xxx\", \"disease\": \"xxx\", \"confidence\": 0.9}, {\"reason\": \"xxx\", \"disease\": \"xxx\", \"confidence\": 0.1}]}\n```\n\nPresent illness: {present_illness}\nPHI findings: {phi_findings}\nPhysical examination: {physical_examination}\nPE findings: {pe_findings}\nAnswer:",
            "keys": ["{candidate_diseases}", "{present_illness}", "{phi_findings}", "{physical_examination}", "{pe_findings}"]
        },
        "return_json": True,
        "format": {"preliminary_diagnosis": list},
        "inp": ["candidate_diseases", "present_illness", "phi_findings", "physical_examination", "pe_findings"],
        "out": {"preliminary_diagnosis": "preliminary_diagnosis"},
        "next": ["FinalDiagnosis"]
    },
    "FinalDiagnosis": {
        "prompt": {
            "prompt": "Based on the patient's present illness, physical examination, laboratory test and imaging test results given below, as well as the preliminary diagnosis, please think carefully and give the final diagnosis results from these diseases {candidate_diseases} and corresponding diagnosis basis, reasons and confidence. Return a structured answer in JSON format, e.g.\n```json\n{\"diagnosis_basis\": \"xxx\", \"diagnosis_reason\": \"xxx\", \"final_diagnosis\": \"xxx\", \"confidence\": 0.xx}\n```\n\nPreliminary diagnosis: {preliminary_diagnosis}\nPresent illness: {present_illness}\nPHI findings: {phi_findings}\nPhysical examination: {physical_examination}\nPE findings: {pe_findings}\nLaboratory tests: {laboratory_tests}\nLab findings: {lab_findings}\nRadiology: {radiology}\nImaging findings: {rad_findings}\nAnswer:",
            "keys": ["{candidate_diseases}", "{preliminary_diagnosis}", "{present_illness}", "{phi_findings}", "{physical_examination}", "{pe_findings}", "{laboratory_tests}", "{lab_findings}", "{radiology}", "{rad_findings}"]
        },
        "return_json": True,
        "format": {"final_diagnosis": str, "confidence": float},
        "inp": ["candidate_diseases", "preliminary_diagnosis", "present_illness", "phi_findings", "physical_examination", "pe_findings", "laboratory_tests", "lab_findings", "radiology", "rad_findings"],
        "out": "diagnosis",
        "next": ["ConfidenceExceedsThreshold"]
    },
    "ConfidenceExceedsThreshold": {
        "inp": ["diagnosis"],
        "use_llm": False,
        "code": "{diagnosis}['confidence']>=0.95",
        "max_cnt": 2,
        "next": {
            True: "exit",
            False: ["SelectReCheckReport", "CopyDiagnosis"]
        }
    },
    "CopyDiagnosis": {
        "out": "pre_diagnosis",
        "item": "diagnosis",
    },
    "SelectReCheckReport": {
        "prompt": {
            "prompt": "The confidence for the final diagnosis is below threshold (0.95). Now you need to re-examine your diagnosis based on the diagnostic criteria of these diseases {candidate_diseases}, as well as the previous preliminary and final diagnoses, please think carefully and select the report from ['present_illness', 'physical_examination', 'laboratory_tests', 'radiology'] that you want to check carefully again and give the corresponding purpose. Return the answer in JSON format, e.g.\n```json\n{\"report\": \"xxx\", \"purpose\": \"xxx\"}\n```\n\nPreliminary diagnosis: {preliminary_diagnosis}\nPrevious final diagnosis: {diagnosis}\nAnswer:",
            "keys": ["{candidate_diseases}", "{preliminary_diagnosis}", "{diagnosis}"]
        },
        "return_json": True,
        "format": {"report": ['present_illness', 'physical_examination', 'laboratory_tests', 'radiology'], "purpose": str},
        "inp": ["candidate_diseases", "preliminary_diagnosis", "diagnosis"],
        "out": "recheck_item",
        "next": ["GetReport", "ReDiagnose"]
    },
    "GetReport": {
        "inp": ["recheck_item"],
        "out": "recheck_report",
        "item": "recheck_item->report",
    },
    "ReDiagnose": {
        "prompt": {
            "prompt": "Based on the rechecked report and the purpose for rechecking, as well as the previous preliminary diagnosis and final diagnosis, please re-evaluate the patient's condition carefully. Take into account the diagnostic criteria and update the final diagnosis (from these diseases {candidate_diseases}) if necessary. Clearly state the basis, reasons, and confidence for your updated diagnosis. Return a structured answer in JSON format, e.g.\n```json\n{\"diagnosis_basis\": \"xxx\", \"diagnosis_reason\": \"xxx\", \"final_diagnosis\": \"xxx\", \"confidence\": 0.xx}\n```\n\nRechecked item: {recheck_item}\nRechecked report details: {recheck_report}\nPreliminary diagnosis: {preliminary_diagnosis}\nPrevious final diagnosis: {pre_diagnosis}\nAnswer:",
            "keys": ["{candidate_diseases}", "{recheck_item}", "{recheck_report}", "{preliminary_diagnosis}", "{pre_diagnosis}"]
        },
        "return_json": True,
        "format": {"final_diagnosis": str, "confidence": float},
        "inp": ["candidate_diseases", "recheck_item", "recheck_report", "preliminary_diagnosis", "pre_diagnosis"],
        "reset_out": "diagnosis",
        "next": ["ConfidenceExceedsThreshold"]
    }
}

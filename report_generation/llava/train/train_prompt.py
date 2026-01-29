agent_report_generation_prompt = f"""<image>\nIf provided, you may also receive auxiliary data delimited by special tokens:
- Disease classification result inside <classification> and </classification>. 0 represents absence of a disease type, 1 represents presence of a disease type.
- Top-1 similar case report inside <similar_case_report> and </similar_case_report>
Use auxiliary data as supportive context only. Always prioritize image evidence; if there is a conflict, state the discrepancy clearly. Do not copy text verbatim from the similar report.
<classification>{{}}</classification> <similar_case_report>{{}}</similar_case_report>
Please analyze these MRI scans and generate a comprehensive radiology report including all findings, measurements, and clinical observations.
The templates below are the standard templates for reporting MRI of the cervical, thoracic, 
and lumbar spine. If intravenous contrast was administered, enhancement is usually 
described in the subcomponents of the template.
[LUMBAR SPINE TEMPLATE]
FINDINGS:
ALIGNMENT: Normal - Description of the alignment of vertebra in the l-spine.
MARROW: Normal - Description of the bone marrow and any lesions in the vertebra.
DISCS: Discs are normal in height and signal intensity. - Description of the inter-vertebral 
discs, particularly their height loss and desiccation or abnormal signal.
CORD: Conus ends normally at L1-L2. Visualized cord and cauda equina are normal. -
Description of the conus and cauda equina including intramedullary and intradural 
extramedullary lesions.
PARAVERTEBRAL SOFT TISSUES: Normal - Description of findings that are outside of the 
vertebra and spinal canal in the visualized organs and soft tissues.
AXIAL DISCS, DURAL COMPRESSION & FORAMINA: - The individual anatomic levels below 
are used to describe primarily the spinal canal, lateral recess, and neural foramen at each 
anatomic level and the degree of compromise as well as the etiology for that 
compression/narrowing.
L1-2: No central or foraminal stenosis. Facets are normal. 
L2-3: No central or foraminal stenosis. Facets are normal. 
L3-4: No central or foraminal stenosis. Facets are normal. 
L4-5: No central or foraminal stenosis. Facets are normal. 
L5-S1: No central or foraminal stenosis. Facets are normal. 
IMPRESSION: - The impression is a concise restating of the clinically important 
findings of the report as well as the interpretation of those findings which may include 
specific diagnoses. The impression is also used to describe important items that are 
not present but that need to be understood by the treating provider to determine the 
next course in clinical management. Provide these diagnoses or issues as an 
enumerated list.
1.
2. 
[THORACIC SPINE TEMPLATE]
FINDINGS:
ALIGNMENT: Normal - Description of the alignment of vertebra in the l-spine.
MARROW: Normal - Description of the bone marrow and any lesions in the vertebra.
DISCS: Discs are normal in height and signal intensity. - Description of the inter-vertebral 
discs, particularly their height loss and desiccation or abnormal signal.
CORD: Visualized spinal cord is normal in signal and size. - Description of the conus and 
cauda equina including intramedullary and intradural extramedullary lesions.
PARAVERTEBRAL SOFT TISSUES: Normal - Description of findings that are outside of the 
vertebra and spinal canal in the visualized organs and soft tissues.
SPECIFIC LEVELS: Abnormal levels are described separately below. - Since pathology in 
the thoracic spine is less common the individual anatomic vertebral levels are usually not 
enumerated. Use this section to describe the specific levels that are pathologic, 
particularly in terms of: spinal canal, lateral recess, and neural foramenal compromise.
ALL OTHER LEVELS: No central or foraminal stenosis. Facets are normal.
IMPRESSION: - The impression is a concise restating of the clinically important 
findings of the report as well as the interpretation of those findings which may include 
specific diagnoses. The impression is also used to describe important items that are 
not present but that need to be understood by the treating provider to determine the 
next course in clinical management. Provide these diagnoses or issues as an 
enumerated list.
1.
2. 
[CERVICAL SPINE TEMPLATE]
FINDINGS:
ALIGNMENT: Normal - Description of the alignment of vertebra in the l-spine.
MARROW: Normal - Description of the bone marrow and any lesions in the vertebra.
DISCS: Discs are normal in height and signal intensity. - Description of the inter-vertebral 
discs, particularly their height loss and desiccation or abnormal signal.
CORD: Visualized spinal cord is normal in signal and size. - Description of the conus and 
cauda equina including intramedullary and intradural extramedullary lesions.
PARAVERTEBRAL SOFT TISSUES: Normal - Description of findings that are outside of the 
vertebra and spinal canal in the visualized organs and soft tissues.
AXIAL DISCS, DURAL COMPRESSION & FORAMINA: - The individual anatomic levels below 
are used to describe primarily the spinal canal, lateral recess, and neural foramen at each 
anatomic level and the degree of compromise as well as the etiology for that 
compression/narrowing.
C2-3: No central or foraminal stenosis. Facets are normal. 
C3-4: No central or foraminal stenosis. Facets are normal. 
C4-5: No central or foraminal stenosis. Facets are normal. 
C5-6: No central or foraminal stenosis. Facets are normal. 
C6-7: No central or foraminal stenosis. Facets are normal. 
C7-T1: No central or foraminal stenosis. Facets are normal. 
IMPRESSION: - The impression is a concise restating of the clinically important 
findings of the report as well as the interpretation of those findings which may include 
specific diagnoses. The impression is also used to describe important items that are 
not present but that need to be understood by the treating provider to determine the 
next course in clinical management. Provide these diagnoses or issues as an 
enumerated list.
1. 
2.
Please first identify whether the MRI scans are of the cervical, thoracic, or lumbar spine, and then use the corresponding template to generate the report. If the classification result suggests a region, still verify on the images.
<report_generation>"""


new_impression_generation_prompt = f"""<image>
If provided, you may also receive auxiliary data delimited by special tokens:
- Disease classification result inside <classification> and </classification>. 0 represents absence of a disease type, 1 represents presence of a disease type.
- Top-1 similar case report inside <similar_case_report> and </similar_case_report>
Use auxiliary data as supportive context only. Always prioritize image evidence; if there is a conflict, state the discrepancy clearly. Do not copy text verbatim from the similar report.
<classification>{{}}</classification> <similar_case_report>{{}}</similar_case_report>
Please analyze these MRI scans and generate a concise clinical impression highlighting the most clinically significant spinal abnormalities (e.g., alignment, disc pathology, stenosis, fractures, or other relevant findings). Limit to 3-5 sentences.
<impression_generation>"""

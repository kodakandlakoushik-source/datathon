SMART CITY — WHERE SHOULD WE FIX FIRST?
Interactive dashboard prototype

FEATURES IMPLEMENTED
- Map view and ranked location table
- Complaint and incident trend charts
- Simple anomaly flags for unusual monthly incident volume
- Explainable priority score using severity, frequency, complaint volume, local context and waiting time
- Adjustable score weights
- Public complaint resolution-time dashboard and overdue queue
- Recommendation queue with reasons, evidence, suggested action and CSV export
- Filters by zone, incident type and severity
- CSV exports for ranked locations and recommendations

DATA
- locations.csv (1,000 locations)
- incidents.csv (10,000 incidents)
- public_complaints.csv (8,000 complaints)
- Shared join key: location_id

IMPORTANT LIMITATIONS
This is a decision-support prototype, not a production or safety-critical system. The dataset and coordinates are synthetic and must not be presented as real city locations. The current score is a transparent heuristic, not a validated risk prediction. The app does not claim a confidence score where no validated model exists. Traffic impact is approximated using available context because no dedicated live traffic measurement field is supplied. Forecasting and resolution-time ML models are not trained in this version.

RUN LOCALLY
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run: pip install -r requirements.txt
4. Run: streamlit run app.py
5. Open the local URL shown in the terminal.

The project is intended for demonstration and should be validated with real, authorized data and stakeholder-approved priorities before operational use.

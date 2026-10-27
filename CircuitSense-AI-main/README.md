# ⚡ CircuitSense AI

**AI-Powered Electronic Circuit Troubleshooting & Learning Assistant for ECE Students**

CircuitSense AI is an intelligent troubleshooting tool designed for Electrical and Computer Engineering (ECE) students and electronics makers. By combining computer vision with deep domain heuristics via **Google Gemma 4 on Ollama Cloud**, the application analyzes breadboard and prototype circuit photographs, flags visual discrepancies against reported symptoms, identifies plausible faults with qualitative confidence ratings, and provides beginner-friendly, safety-first verification procedures.

---

## 📑 Table of Contents

1. [Problem Statement & Motivation](#-problem-statement--motivation)
2. [Current Implemented Features](#-current-implemented-features)
3. [How CircuitSense AI Works](#-how-circuitsense-ai-works)
4. [System Architecture & Data Flow](#-system-architecture--data-flow)
5. [Technology Stack](#-technology-stack)
6. [Project Directory Structure](#-project-directory-structure)
7. [Prerequisites & Supported Systems](#-prerequisites--supported-systems)
8. [macOS Setup Guide](#-macos-setup-guide)
9. [Windows Setup Guide (PowerShell)](#-windows-setup-guide-powershell)
10. [Ollama Cloud Authentication & Model Verification](#-ollama-cloud-authentication--model-verification)
11. [Launching & Using the Application](#-launching--using-the-application)
12. [Running Automated Tests & Code Verification](#-running-automated-tests--code-verification)
13. [Illustrative Example Walkthrough](#-illustrative-example-walkthrough)
14. [Troubleshooting Guide](#-troubleshooting-guide)
15. [Safety, Limitations & Responsible Use](#-safety-limitations--responsible-use)
16. [Git Collaboration & Hackathon Handoff Workflow](#-git-collaboration--hackathon-handoff-workflow)
17. [Current Implementation Status & Roadmap](#-current-implementation-status--roadmap)
18. [Teammate Quick-Start Checklist](#-teammate-quick-start-checklist)

---

## 🎯 Problem Statement & Motivation

Prototyping circuits on solderless breadboards is a foundational rite of passage for every engineering student, but it is notoriously prone to subtle, hard-to-spot failure modes:
- **Hidden Connectivity Issues:** Solderless breadboards feature internal spring clips and split power distribution rails that are invisible from above.
- **Orientation & Polarity Traps:** Diodes, LEDs, ICs, and electrolytic capacitors fail silently or burn out when reversed.
- **Deceptive Visuals:** A jumper wire seated in an adjacent tie-point row or an unverified resistor color code can lead to hours of frustrating debugging.
- **Safety Hazards:** Inexperienced students frequently attempt unsafe measurements—such as probing resistance on a live circuit (destroying the DMM) or placing an ammeter across a battery (causing a direct short circuit).

**CircuitSense AI** acts as a virtual lab teaching assistant. It does not blindly guess repairs or fabricate component ratings; instead, it observes physical component placement, checks for contradictions between visual evidence and reported symptoms, outlines unknowns, explains the core electronics concept, and guides the student through safe, step-by-step verification steps.

---

## ✨ Current Implemented Features

- **Multimodal Visual Inspection:** Ingests breadboard photographs (`PNG`, `JPEG`, `WebP` up to 10 MB) and generates immediate in-browser previews.
- **Symptom & Context Capture:** Accepts student problem descriptions (up to 2,000 characters) alongside optional circuit category and component specifications (up to 500 characters).
- **Contradiction & Discrepancy Detection:** Flags conflicts between photographic evidence (e.g., photo shows an illuminated LED) and student descriptions (e.g., "LED will not glow") without inventing unsupported timeline narratives.
- **Non-Forced Fault Hypotheses:** Generates up to three plausible candidate faults ordered by relevance. If the circuit appears correctly wired or evidence is contradictory, returns an empty fault list (`[]`) rather than inventing non-existent bugs.
- **Qualitative Confidence Ratings:** Rates hypotheses strictly as `Low`, `Medium`, or `High` based on visible evidence; never outputs fabricated percentages or false certainty.
- **Uncertainty & Unknowns Isolation:** Explicitly enumerates information that cannot be determined visually (e.g., unreadable resistor color bands, internal rail breaks, unverified power supply voltages).
- **Educational Theory Explanations:** Explains the underlying physical concept (e.g., diode forward bias, Ohm's law, voltage drop) in encouraging, beginner-friendly language.
- **Strict Multimeter Safety Protocols:**
  - Instructs disconnecting power before altering wiring or measuring resistance.
  - Explains lifting one component lead to prevent parallel sneak paths from altering resistance readings.
  - Prohibits powered voltage measurements unless justified by a specific troubleshooting question.
  - Mandates explicit DC voltage settings and parallel probe placement.
  - Forbids measuring resistance on live circuits and measuring current directly across power supplies.
- **JSON Report Export:** One-click download of the structured diagnostic results (`circuitsense_report.json`) for lab reports and peer review.
- **Comprehensive Offline Test Suite:** 41 deterministic automated tests with mocked API interactions and zero external network calls.

---

## 🔬 How CircuitSense AI Works

1. **Upload & Preview:** The student uploads an image of their physical circuit. Pillow verifies byte integrity, file size, and image format (`PNG`, `JPEG`, `WebP`).
2. **Context Input:** The student inputs their observed symptom (e.g., "Output drops to 0V when switch is toggled"), optional circuit category, and known parts.
3. **Payload Construction:** The client Base64-encodes the image and structures a system prompt enforcing ECE diagnostic principles alongside a strict JSON response schema.
4. **Cloud Multimodal Inference:** Dispatches an HTTP POST request to the local Ollama gateway (`/api/chat`), which routes the multimodal request to the **Gemma 4 Cloud** model (`gemma4:cloud`).
5. **Defensive Parsing & Validation:** The client unpacks the assistant response, strips markdown code fences if present, and validates that the payload is a valid JSON object.
6. **Structured Rendering:** Streamlit dynamically renders the assessment summary, safety alerts, candidate fault expander cards, unknowns list, action steps, theory concepts, and download button.

---

## 🏛️ System Architecture & Data Flow

```text
+-----------------------------------------------------------------------------------+
|                                 USER BROWSER                                      |
|  - Uploads photo (PNG, JPEG, WebP <= 10 MB)                                       |
|  - Inputs symptom description (<= 2000 chars) & optional circuit metadata         |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         STREAMLIT APPLICATION (app.py)                            |
|  - Renders UI, image preview, and input forms                                     |
|  - Performs client-side validations                                               |
|  - Manages session state to prevent accidental duplicate inference calls          |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         GEMMA CLIENT (gemma_client.py)                           |
|  - Validates text limits & checks image headers using Pillow (PIL)                |
|  - Encodes image bytes to Base64                                                  |
|  - Merges DIAGNOSTIC_SYSTEM_PROMPT & formatted user prompt from prompts.py        |
|  - Configures 180s finite HTTP timeout & structured exception handling            |
+-----------------------------------------+-----------------------------------------+
                                          |
                 HTTP POST (JSON)         |  Endpoint: http://localhost:11434/api/chat
                                          v
+-----------------------------------------------------------------------------------+
|                             OLLAMA LOCAL GATEWAY                                  |
|  - Listens on port 11434                                                          |
|  - Handles authenticated session token (via 'ollama signin')                      |
|  - Dispatches multimodal inference payload to Ollama Cloud                        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        GOOGLE GEMMA 4 CLOUD (gemma4:cloud)                        |
|  - 32.7B Multimodal Vision Model with Thinking & Tool Capabilities                |
|  - Inspects circuit image + user text prompt                                      |
|  - Emits JSON strictly adhering to DIAGNOSTIC_SCHEMA                              |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     PARSING, DEFENSIVE RENDERING & EXPORT                         |
|  - gemma_client.py parses string envelope and verifies dictionary structure       |
|  - app.py renders Summary, Hazards, Fault Cards, Unknowns, and Theory             |
|  - Generates downloadable formatted report: circuitsense_report.json              |
+-----------------------------------------------------------------------------------+
```

---

## 💻 Technology Stack

| Layer | Technology | Version | Purpose in CircuitSense AI |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11 – 3.13 | Core programming language for client, prompts, and application. |
| **Web Framework** | Streamlit | 1.66.1 | Web interface, responsive two-column layout, and state management. |
| **Image Processing** | Pillow (PIL) | 13.0.0 | Binary image validation, format detection (`PNG`, `JPEG`, `WEBP`), and integrity checks. |
| **HTTP Client** | Requests | 2.34.2 | Communicates with the Ollama local daemon (`/api/chat`) with configurable timeouts. |
| **AI Runtime** | Ollama | Latest | Local daemon managing authentication and routing requests to Ollama Cloud models. |
| **Diagnostic Model** | Gemma 4 Cloud | `gemma4:cloud` (32.7B) | Multimodal vision model performing circuit feature analysis and fault hypothesis generation. |
| **Test Framework** | Pytest | 9.1.1 | Automated offline unit and regression test suite (41 tests). |

---

## 📂 Project Directory Structure

```text
circuitsense-ai/
├── app.py                  # Streamlit application UI, input forms, and defensive results rendering
├── gemma_client.py         # Client module: input validation, Base64 encoding, Ollama API caller, response parser
├── prompts.py              # Diagnostic system prompt, JSON response schema, and user prompt builder
├── requirements.txt        # Pinned project dependencies
├── circuit.png             # Sample test image (9V battery, breadboard, resistor, LED, jumper wires)
├── README.md               # Complete project documentation and handoff guide
├── .gitignore              # Git exclusions for virtual environments, caches, bytecode, and secrets
└── tests/
    └── test_inputs.py      # 41 automated unit tests (validations, mocked responses, prompt rules)
```

---

## ⚙️ Prerequisites & Supported Systems

- **Operating Systems:**
  - **macOS:** Apple Silicon (M1/M2/M3/M4) or Intel, macOS 12 Monterey or later.
  - **Windows:** Windows 10 or Windows 11 (64-bit), via **PowerShell**.
  - **Linux:** Ubuntu 22.04+ or equivalent modern distribution.
- **Python:** Version 3.11, 3.12, or 3.13 installed and accessible on PATH.
- **Ollama:** Installed from official distribution with an active Ollama account.
- **Network Connection:** Active internet connection required (for Ollama Cloud inference).

---

## 🍎 macOS Setup Guide

Open **Terminal** (`Applications` > `Utilities` > `Terminal`) and complete these steps:

### 1. Verify Prerequisites
```bash
python3 --version
python3 -m pip --version
```
*If Python 3 is not installed, download the official installer from [python.org](https://www.python.org/downloads/) or install via Homebrew (`brew install python`).*

### 2. Install Ollama
Download and install the official macOS app from [ollama.com/download/mac](https://ollama.com/download/mac), or install via Homebrew:
```bash
brew install ollama
```
Verify the installation:
```bash
ollama --version
```
*Note: If you installed the Ollama macOS desktop app, it runs automatically in the background menu bar. If running via terminal, execute `ollama serve` in a separate tab.*

### 3. Clone Repository & Navigate
```bash
git clone <PRIVATE_DEVELOPMENT_REPO_URL>
cd circuitsense-ai
```

### 4. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Authenticate to Ollama Cloud
Sign in using your authorized Ollama account:
```bash
ollama signin
```
*Follow the browser prompt to complete authentication.*

### 7. Verify Model Access
Run a fast, lightweight text test to confirm cloud authorization:
```bash
ollama run gemma4:cloud "Confirm connection to CircuitSense AI"
```
*(This confirms that your Ollama Cloud account is authorized. Multimodal image diagnosis runs through the Python app).*

### 8. Run Automated Tests
```bash
python -m pytest -v
```
*(Confirm all 41 unit tests pass).*

### 9. Launch the Streamlit App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🪟 Windows Setup Guide (PowerShell)

Open **PowerShell** (press `Win + X` and select **Windows PowerShell** or **Terminal**) and complete these steps:

### 1. Verify Prerequisites
```powershell
python --version
pip --version
```
*If Python is missing, download Python 3.11+ from [python.org/downloads](https://www.python.org/downloads/). During installation, **make sure to check "Add python.exe to PATH"**.*

### 2. Install Ollama for Windows
Download and run the official Windows installer from [ollama.com/download/windows](https://ollama.com/download/windows).
Verify in PowerShell:
```powershell
ollama --version
```
*Note: The Ollama Windows installer automatically starts the Ollama background service in your Windows System Tray. If it is stopped, launch Ollama from your Windows Start Menu.*

### 3. Clone Repository & Navigate
```powershell
git clone <PRIVATE_DEVELOPMENT_REPO_URL>
cd circuitsense-ai
```

### 4. Create Virtual Environment
```powershell
python -m venv .venv
```

### 5. Activate Virtual Environment (or Safe Fallback)
Try activating the virtual environment in PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

**If PowerShell blocks script execution** with an `ExecutionPolicy` error (`PSSecurityException`), do not perform a machine-wide policy change. Instead, use either of these safe alternatives:

- **Alternative A (Process-Scoped Bypass):**
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\.venv\Scripts\Activate.ps1
  ```
- **Alternative B (Direct Python Binary Call):**
  You can execute commands directly through the virtual environment's Python executable without activating scripts:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  ```

### 6. Install Dependencies
If activated:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 7. Authenticate to Ollama Cloud
```powershell
ollama signin
```
*Follow the browser instructions to log into your authorized Ollama account.*

### 8. Verify Model Access
```powershell
ollama run gemma4:cloud "Confirm connection to CircuitSense AI"
```

### 9. Run Tests & Code Compilation Check
If activated:
```powershell
python -m pytest -v
python -m py_compile app.py prompts.py gemma_client.py
```
Or using direct binary paths:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m py_compile app.py prompts.py gemma_client.py
```

### 10. Launch Streamlit
If activated:
```powershell
streamlit run app.py
```
Or using direct binary paths:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## ☁️ Ollama Cloud Authentication & Model Verification

CircuitSense AI leverages **Gemma 4 Cloud (`gemma4:cloud`)**, a 32.7B parameter multimodal model hosted remotely on Ollama Cloud infrastructure. This architecture offers two major advantages:
1. **Zero Massive Weights Download:** Teammates do not need to download 20+ GB of model weights to local storage.
2. **Lightweight Hardware Footprint:** Runs seamlessly on laptops without dedicated discrete GPUs.

### How to Authenticate
Run the signin command in your terminal:
```bash
ollama signin
```
This opens your browser to `https://ollama.com` where you log in with your authorized account.

### How to Verify Access
Run a short text query:
```bash
ollama run gemma4:cloud "Hello Gemma, please confirm cloud connectivity."
```
> **Important Note:** A successful text response confirms that your Ollama Cloud account is authenticated and reachable. It verifies cloud gateway communication; full image-based circuit troubleshooting is tested by running the Streamlit app with an actual image.

---

## 🖥️ Launching & Using the Application

### 1. Start the App
```bash
streamlit run app.py
```

### 2. Step-by-Step Usage
1. **Upload Circuit Photo (Required):** Select a clear photograph (`PNG`, `JPEG`, or `WebP`, <= 10 MB). The app displays an instant preview. You can use the included `circuit.png` sample image.
2. **Describe Observed Symptoms (Required):** Enter what is happening vs. what you expected in the text area (e.g., *"The red LED does not turn on when the 9V battery is attached to the breadboard rails"*).
3. **Select Circuit Type (Optional):** Choose from categories like *LED circuit*, *Power supply*, *Amplifier*, *Digital logic*, *Sensor circuit*, *Motor driver*, or *Other*.
4. **Specify Known Components (Optional):** Add component values if known (e.g., *"9V Duracell battery, 330 ohm resistor, 5mm red LED"*).
5. **Click "⚡ Diagnose Circuit":** A genuine spinner indicates analysis progress while Gemma 4 processes the image and text.

### 3. Understanding the Results
- **Assessment Summary:** Concise overview of visible circuit state and symptom correlation.
- **Safety Precautions:** Highlighted precautions; dangerous hazards (e.g., smoke, high voltage, bulging batteries) receive critical callouts.
- **Potential Faults:** Expandable cards displaying up to 3 candidate hypotheses with qualitative confidence (`High`, `Medium`, `Low`), supporting evidence, and safe verification steps.
- **Unknowns & Unconfirmed Details:** Explicit listing of items that cannot be determined visually (e.g., resistor values, hidden breadboard continuity).
- **Recommended Safe Actions:** Ordered sequence of non-powered checks.
- **Electronics Learning Concept:** Clear explanation of the relevant electrical theory for students.
- **Download Diagnostic Report:** Click to save `circuitsense_report.json` to your local machine.

---

## 🧪 Running Automated Tests & Code Verification

The repository contains an automated test suite in `tests/test_inputs.py`. All tests are **100% offline**, utilizing mocked HTTP responses so they can run without network access or active Ollama daemons.

### Run Unit Tests
```bash
python -m pytest -v
```

### Test Coverage Summary (41 Passing Tests)
- **Text Validation:** Empty strings, whitespace-only descriptions, 2000-char limits, 500-char limits on optional fields, non-string type rejection.
- **Image Validation:** Zero-byte files, non-byte data, corrupt headers, files > 10 MB, rejection of unsupported formats (GIF, BMP, TIFF), acceptance of PNG, JPEG, and WebP.
- **Prompt Engineering & Schemas:** Schema completeness, system prompt constraints, user prompt assembly, discrepancy handling, prohibition of false continuity claims, multimeter safety constraints, voltage measurement restrictions.
- **Response Parsing:** Envelope extraction, markdown code block stripping, malformed JSON rejection, non-dictionary JSON rejection, empty possible_faults list handling.
- **Client & Error Handling:** Successful mock diagnosis, environment variable overrides (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`), HTTP 500 handling, timeout handling (180s), connection failure handling, missing content handling, pre-network validation stops.

### Code Compilation Check
Verify syntax across all core modules:
```bash
python -m py_compile app.py prompts.py gemma_client.py tests/test_inputs.py
```

---

## 🔍 Illustrative Example Walkthrough

*(The following walkthrough is based on the included `circuit.png` sample image and is provided for illustration; actual model outputs may vary).*

### Sample Input
- **Image:** [circuit.png](file:///Users/jusujan/circuitsense-ai/circuit.png) (Breadboard with 9V Duracell battery, current-limiting resistor, illuminated red LED, red and black jumper wires).
- **Reported Problem:** *"The red LED does not turn on when the battery is attached to the rails."*
- **Circuit Type:** *LED circuit*
- **Components:** *9V battery, resistor with brown-black-red bands, 5mm red LED*

### Expected Diagnostic Output Structure
- **Discrepancy Note in Summary:** The model notes that while the user reports the LED does not turn on, the photograph shows an illuminated red LED. It notes the visual conflict without inventing unverified timeline explanations (e.g., without assuming the battery died later).
- **Confirmed Faults List:** Returns an empty candidate faults list (`"possible_faults": []`) because the visual evidence contradicts the reported fault and no independent failure is established.
- **Unknowns Identified:**
  - Exact resistance value (color bands visible but require physical DMM measurement to verify).
  - Internal continuity of the breadboard tie points.
  - Present electrical state of the circuit (since the photo is a snapshot in time).
- **Safe Next Steps:**
  1. Confirm the present operating state of the circuit (verify if the LED is currently unlit or lit).
  2. Disconnect the 9V battery before touching components or adjusting jumper wires.
  3. If the LED is dark now, inspect whether jumper leads have slipped from the breadboard tie points.
- **Learning Concept:** Explains forward bias in semiconductor diodes and the necessity of series current-limiting resistors to prevent thermal runaway when powered by a 9V source.
- **Safety Warnings:** Emphasizes disconnecting power before touching components and warns against shorting battery terminals directly.

---

## 🛠️ Troubleshooting Guide

| Symptom | Probable Cause | Action / Resolution |
| :--- | :--- | :--- |
| **`Connection Error: Could not connect to Ollama at http://localhost:11434`** | The Ollama local service daemon is not running. | **macOS:** Launch the Ollama app or run `ollama serve` in Terminal.<br>**Windows:** Launch Ollama from the Start Menu or check the System Tray icon. |
| **`Model Response Error: model 'gemma4:cloud' not found`** | Cloud account not signed in or model access not granted. | Run `ollama signin` in your terminal, sign in with your authorized account, then execute `ollama run gemma4:cloud "test"`. |
| **`Request Timeout: The model took longer than 180 seconds to respond`** | Slow internet connection or high cloud model latency during peak hours. | Ensure your internet connection is active. Retry the submission; large images may take 15–35 seconds. |
| **`Validation Error: Image size exceeds the maximum limit of 10 MB`** | High-resolution phone camera photo exceeds 10 MB. | Crop, compress, or resize the image to under 10 MB before uploading. |
| **`Unsupported image format: 'GIF'`** | File uploaded is not an accepted static format. | Convert the image to PNG, JPEG, or WebP. |
| **Windows: `Activate.ps1 cannot be loaded because running scripts is disabled`** | PowerShell execution policy restrictions. | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` or call `.\.venv\Scripts\python.exe` directly. |
| **`Address already in use: 8501`** | Another Streamlit server instance is running. | Specify an alternative port: `streamlit run app.py --server.port 8502`. |

---

## ⚠️ Safety, Limitations & Responsible Use

### Physical Circuit & Electrical Safety
1. **Always Isolate Power First:** Disconnect the battery or power supply before inspecting wires, moving components, or inserting meter probes.
2. **Never Measure Resistance on Live Circuits:** Measuring resistance on an energized circuit will produce invalid readings and can blow internal multimeter fuses or destroy the instrument.
3. **Never Place an Ammeter Across Power Rails:** Current measurements must always be placed *in series* with a load. Placing multimeter probes in current mode across a battery causes a dead short circuit, resulting in damaged meters, ruptured batteries, or fire hazards.
4. **Critical Hazard Response:** If you detect burning odors, smoke, hot ICs, or swollen battery casings, **immediately disconnect power** and alert your laboratory instructor.

### AI & Vision Model Limitations
- **Educational Hypotheses Only:** CircuitSense AI generates educational troubleshooting hypotheses based on visual evidence. It does not provide certified repair instructions or replacement guarantees.
- **Photographs Cannot Prove Continuity:** An image cannot verify electrical continuity through hidden breadboard spring clips, cold solder joints, or split bus rails. All circuit connections must be verified with safe, de-energized DMM checks.
- **Privacy Assurance:** The application runs locally via the Ollama gateway and does not store user photos, credentials, or Base64 data in public logs or cloud databases.

---

## 🤝 Git Collaboration & Hackathon Handoff Workflow

When collaborating during the hackathon, follow this clean-transfer process between repositories:

### Repository Roles
- **`<PRIVATE_DEVELOPMENT_REPO_URL>`:** The team's internal working repository where development, testing, and debugging take place.
- **`<EVENT_PRIVATE_REPO_URL>`:** The official private submission repository allocated by hackathon organizers.

### Clean Transfer Procedure (No Dirty History or Git Conflicts)
Do **not** force-push or copy the `.git` folder from your development repository into the event repository. Follow these steps:

1. **Clone the Event Submission Repository:**
   ```bash
   git clone <EVENT_PRIVATE_REPO_URL> hackathon-submission
   cd hackathon-submission
   ```

2. **Copy Project Files from Development Directory:**
   Copy only the application files into `hackathon-submission`:
   - `app.py`
   - `gemma_client.py`
   - `prompts.py`
   - `requirements.txt`
   - `circuit.png`
   - `README.md`
   - `.gitignore`
   - `tests/` directory
   *(Do **NOT** copy `.venv/`, `.git/`, `.pytest_cache/`, or `__pycache__/`).*

3. **Verify Everything in the Event Repo:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m pytest -v
   python -m py_compile app.py prompts.py gemma_client.py
   ```

4. **Commit & Push to the Submission Repository:**
   ```bash
   git status
   git add app.py gemma_client.py prompts.py requirements.txt circuit.png README.md .gitignore tests/
   git commit -m "feat: complete CircuitSense AI hackathon MVP release"
   git push origin main
   ```

---

## 🗺️ Current Implementation Status & Roadmap

### Currently Implemented (MVP)
- [x] Streamlit web UI with image upload, preview, and responsive cards.
- [x] Gemma 4 Cloud vision integration via Ollama `/api/chat`.
- [x] Strict input boundaries (10 MB image, 2000-char problem, 500-char optional metadata).
- [x] Defensive model JSON extraction and markdown fence stripping.
- [x] Discrepancy flagging (contradiction handling).
- [x] Non-forced fault hypothesis generation (permits empty fault lists).
- [x] Rigorous multimeter safety rules (power isolation, DC voltage configuration).
- [x] Downloadable JSON diagnostic report.
- [x] 41 automated Pytest unit tests (100% passing offline).

### Future Roadmap (Post-Hackathon)
- [ ] **Interactive Schematic Diagram Generator:** Render companion SVG schematics alongside breadboard photos.
- [ ] **SPICE Simulation Verification:** Connect to PySpice / ngspice to simulate expected voltage nodes before recommending DMM checks.
- [ ] **Multi-Angle Capture:** Allow uploading multiple photos (top-down, profile, component macro) to resolve hidden tie-point ambiguities.
- [ ] **Component Markings OCR Lookup:** Dedicated optical character recognition for IC part numbers (e.g., NE555, LM358, 74HC00).

---

## ⚡ Teammate Quick-Start Checklist

Need to run CircuitSense AI in under 2 minutes? Follow these steps:

```bash
# 1. Clone & enter repository
git clone <PRIVATE_DEVELOPMENT_REPO_URL>
cd circuitsense-ai

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Sign in to Ollama Cloud
ollama signin

# 5. Verify tests & launch application
python -m pytest -v
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser and upload **`circuit.png`** to test!

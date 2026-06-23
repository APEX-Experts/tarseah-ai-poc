# Tarseah AI — Developer Documentation

> **Quick reference for frontend engineers and contributors.**

---

## 1. Project Overview

Tarseah AI is a **proposal generation engine** tailored for the Saudi/Gulf government tender market. Given a company's profile and a tender's RFP documents, it automatically drafts all 11 standard sections of a technical proposal using an LLM (currently Groq — `openai/gpt-oss-20b`).

The flow is simple:
1. **Initialize** — upload project documents and extract their text once.
2. **Generate** — call one section at a time; each result is persisted server-side.
3. **Read** — fetch any completed section whenever you need it.

Output language is configurable per call: **Arabic (`ar`, default) or English (`en`)**.

---

## 2. API Reference

Base URL (local dev): `http://localhost:5000`

All proposal endpoints are prefixed with `/proposals`.

---

### 2.1 Initialize Project — `POST /proposals/initialize/{project_id}`

**Purpose:** Upload project documents and run the context initializer. This must be called **once before any section can be generated**. It extracts text from all uploaded files and saves it to a `shared_memory.json` file on the server.

**Path parameter:**
| Param | Type | Description |
|---|---|---|
| `project_id` | string | Unique identifier for the project (alphanumeric, dashes, underscores). |

**Query parameter:**
| Param | Type | Default | Description |
|---|---|---|---|
| `force_reset` | bool | `false` | If `true`, wipes and re-initializes a previously initialized project. |

**Form fields (multipart/form-data):**

You can upload files in two ways:

_Option A — Generic upload (filename must contain a keyword):_
| Field | Description |
|---|---|
| `file` | One or more files (keyword-based auto-routing). |
| `files` | Alias for `file`; same behavior. |

_Option B — Explicit role upload:_
| Field | Role |
|---|---|
| `tender_file` | Tender RFP document |
| `company_profile` | Company Profile / Experience document |
| `bid_details` | Specific Bid details document |

> ⚠️ **If using Option A**, the filename must contain the role keyword (e.g. `tender`, `rfp`, `company`, `profile`, `bid`, `details`). See Section 3 for naming conventions.

**Success response (`200`):**
```json
{
  "status": "success",
  "message": "2 file(s) uploaded and context initialized.",
  "project_id": "my-project",
  "uploaded_files": [...],
  "shared_memory_path": "/abs/path/to/shared_memory.json",
  "node_output": {
    "tender_text_length": 84200,
    "company_assets_text_length": 31000,
    "bid_details_text_length": 5100,
    "sections_initialized": ["cover_letter", "executive_summary", ...]
  }
}
```

---

### 2.2 Generate Section (blocking) — `POST /proposals/generate/{project_id}/{section_type}?language=ar`

**Purpose:** Generate a single proposal section. The server calls the LLM and returns the **full generated Markdown** when complete. Use this when you don't need streaming. If query parameter `language` is omitted, it defaults to Arabic (`ar`).

**Path parameters:**
| Param | Description |
|---|---|
| `project_id` | Must match an already-initialized project. |
| `section_type` | One of the 11 section keys listed below. |

**Query parameters:**
| Param | Type | Default | Description |
|---|---|---|---|
| `language` | `"ar"` \| `"en"` | `"ar"` | Output language (Defaults to `"ar"` if not specified). |

**Valid `section_type` values:**

| Key | Arabic Name | English Name |
|---|---|---|
| `cover_letter` | خطاب التقديم | Cover Letter |
| `executive_summary` | الملخص التنفيذي | Executive Summary |
| `scope_understanding` | فهم نطاق العمل | Understanding of Scope of Work |
| `vision_2030` | التوافق مع رؤية 2030 | Vision 2030 Alignment |
| `company_profile` | ملف الشركة | Company Profile |
| `past_projects` | المشاريع السابقة | Past Projects (Track Record) |
| `methodology` | منهجية التنفيذ | Execution Methodology |
| `team` | هيكل فريق العمل | Project Team and Structure |
| `timeline` | الجدول الزمني | Project Timeline and Schedule |
| `quality_and_risk` | إدارة الجودة والمخاطر | Quality and Risk Management |
| `pricing` | العرض المالي والتسعير | Financial Proposal and Pricing |

**Success response (`200`):**

#### Arabic Output (Default: if no parameter or `language=ar` specified)
```json
{
  "status": "success",
  "project_id": "my-project",
  "section_type": "methodology",
  "section_config_type": "dynamic",
  "generated_markdown": "## منهجية التنفيذ\n\nتعتمد منهجية تنفيذ هذا المشروع على تقديم حلول متكاملة تضمن تحقيق أعلى مستويات الكفاءة والجودة...",
  "input_tokens": 18340,
  "output_tokens": 2100,
  "was_truncated": false,
  "finish_reason": "stop",
  "sections_progress": {
    "cover_letter": { "status": "DRAFT", "has_content": true },
    "methodology": { "status": "DRAFT", "has_content": true }
  }
}
```

#### English Output (`language=en`)
```json
{
  "status": "success",
  "project_id": "my-project",
  "section_type": "methodology",
  "section_config_type": "dynamic",
  "generated_markdown": "## Execution Methodology\n\nThe project execution methodology relies on delivering integrated solutions that ensure the highest levels of efficiency and quality...",
  "input_tokens": 17850,
  "output_tokens": 1950,
  "was_truncated": false,
  "finish_reason": "stop",
  "sections_progress": {
    "cover_letter": { "status": "DRAFT", "has_content": true },
    "methodology": { "status": "DRAFT", "has_content": true }
  }
}
```

> ⚠️ Check `was_truncated`. If `true`, the model hit its output token limit and the content may be cut off.

---

### 2.3 Generate Section (streaming) — `POST /proposals/generate/{project_id}/{section_type}/stream?language=ar`

**Purpose:** Same as 2.2 but returns the content as a **Server-Sent Events (SSE)** stream. Use this to display the text as it is being written in real-time. Defaults to Arabic (`ar`) if no parameter is provided.

Also supports `GET /proposals/generate/{project_id}/{section_type}/stream?language=ar` (same URL, same params) for easier browser-side `EventSource` usage.

**Headers returned:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

**SSE event format:**

Each event is a JSON object on a `data:` line:

#### Arabic Stream Example (`language=ar`)
```
data: {"chunk": "## منهجية "}

data: {"chunk": "التنفيذ\n\nتعتمد "}

data: {"chunk": "منهجية تنفيذ هذا "}

data: {"done": true, "section_type": "methodology", "input_tokens": 18340, "output_tokens": 2100, "finish_reason": "stop"}

data: [DONE]
```

#### English Stream Example (`language=en`)
```
data: {"chunk": "## Execution "}

data: {"chunk": "Methodology\n\nThe execution "}

data: {"chunk": "methodology relies on "}

data: {"done": true, "section_type": "methodology", "input_tokens": 17850, "output_tokens": 1950, "finish_reason": "stop"}

data: [DONE]
```

| Event | Meaning |
|---|---|
| `{"chunk": "..."}` | Partial Markdown text — append to display buffer. |
| `{"done": true, ...}` | Stream finished; includes final metadata. |
| `{"error": "..."}` | Something went wrong; abort and show error. |
| `[DONE]` | Stream terminator — close the connection. |

---

### 2.4 Get All Sections — `GET /proposals/sections/{project_id}`

**Purpose:** Get a summary of all 11 sections for a project — their status and whether content exists. Useful for rendering a progress dashboard or checking which sections are ready.

**Success response (`200`):**
```json
{
  "status": "success",
  "project_id": "my-project",
  "metadata": { ... },
  "sections": {
    "cover_letter":       { "status": "DRAFT", "has_content": true,  "content_length": 1820 },
    "executive_summary":  { "status": "EMPTY", "has_content": false, "content_length": 0 },
    ...
  }
}
```

**Section statuses:**
| Status | Meaning |
|---|---|
| `EMPTY` | Not yet generated. |
| `DRAFT` | Generated by AI; pending review. |
| `REVIEW` | Under manual review (set externally). |
| `FINAL` | Approved and finalized (set externally). |

---

### 2.5 Get Single Section Content — `GET /proposals/sections/{project_id}/{section_type}`

**Purpose:** Retrieve the full generated Markdown content for one specific section.

**Success response (`200`):**
```json
{
  "status": "success",
  "project_id": "my-project",
  "section_type": "methodology",
  "section_status": "DRAFT",
  "content": "## منهجية التنفيذ\n...",
  "content_length": 4350
}
```

---

### 2.6 Stream Stored Section — `GET /proposals/sections/{project_id}/{section_type}/stream`

**Purpose:** Stream **already-generated** section content using the same SSE format as 2.3. Use this to re-display content with a typing animation without re-calling the LLM.

Returns the same event format as 2.3 (`chunk` → `done` → `[DONE]`).

Also supports `POST`.

---

## 3. File Naming Conventions

When using **generic upload** (`file` / `files` fields), the system routes each file to the correct role by scanning for keywords in the filename. Use the following standard names:

| Filename | Role | Keywords matched |
|---|---|---|
| `tender-rfp.pdf` | Tender / RFP document | `tender`, `rfp` |
| `company-profile.pdf` | Company Profile & Experience | `company`, `profile`, `experience` |
| `bid-details.md` | Bid-specific details & supplements | `bid`, `details` |

### What each file should contain

**`tender-rfp.pdf`**
The official Request for Proposal (RFP) issued by the client. This is the primary source for the project scope, technical requirements, deliverables, timelines, evaluation criteria, and compliance terms. The AI reads this to understand *what* needs to be delivered.

**`company-profile.pdf`**
The bidding company's corporate profile. Should include company history, core services, organizational structure, key personnel, past projects with client names and values, and any certifications. The AI reads this as the *only* source of company facts — it will not invent anything not found here.

**`bid-details.md`**
A supplementary Markdown or text file containing bid-specific context that is not in the RFP or company profile. This is typically used for internal notes, pricing hints, strategic priorities, or client-specific adjustments the team wants to inject into the proposal. It is the primary source for `executive_summary`, `scope_understanding`, `methodology`, `timeline`, and `pricing` sections.

> **Important:** Do not combine files. Keep each role in a separate file with its keyword clearly in the filename so the context initializer can route correctly.

---

## 4. Frontend AI Integration Notes

### How generation works internally

Each section is generated in a **separate, independent LLM call**. The system maintains consistency by injecting previously generated sections as a "compiled memory" block into each new prompt, so later sections stay coherent with earlier ones.

### Recommended generation order

The sections are designed to be generated in this order (each one can reference those before it):

```
cover_letter → executive_summary → scope_understanding → vision_2030
→ company_profile → past_projects → methodology → team
→ timeline → quality_and_risk → pricing
```

Generating out of order is allowed but may reduce consistency.

### Language parameter

Every generate call accepts `?language=ar` (default) or `?language=en`. Arabic uses professional Modern Standard Arabic (فصحى). Set this once per project and use it consistently.

### Handling the `pricing` section

If no pricing data exists in any of the uploaded documents, the `pricing` section will be generated with **all financial fields intentionally left blank**. This is by design — the AI is strictly forbidden from inventing prices. The frontend should display a clear "fill manually" indicator when `pricing` content contains blank table cells.

### Token limits & truncation

Large sections (`scope_understanding`, `methodology`, `timeline`, `pricing`) can be expensive. The system already applies context filtering to reduce token usage. However, if `was_truncated: true` appears in the response, display a warning to the user and allow them to retry.

### SSE integration tips

```javascript
// Using EventSource (GET only)
const source = new EventSource(
  `/proposals/generate/${projectId}/${section}/stream?language=ar`
);

source.onmessage = (event) => {
  if (event.data === "[DONE]") { source.close(); return; }
  const payload = JSON.parse(event.data);
  if (payload.chunk)  appendToEditor(payload.chunk);
  if (payload.done)   saveFinalMetadata(payload);
  if (payload.error)  showError(payload.error);
};

// Using fetch + ReadableStream (POST — recommended for flexibility)
const res = await fetch(`/proposals/generate/${projectId}/${section}/stream?language=ar`, {
  method: "POST"
});
const reader = res.body.getReader();
// ... read chunks and parse SSE lines manually
```

### Project ID rules

Project IDs must contain only alphanumeric characters, dashes (`-`), and underscores (`_`). Use a UUID or a slugified tender name (e.g., `tender-2025-riyadh-metro`).
